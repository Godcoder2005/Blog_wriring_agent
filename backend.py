# --- All the imports we need to make this thing work ---
from __future__ import annotations
import operator
import os
import sqlite3
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated, Dict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# Load our API keys and secrets from the .env file
load_dotenv()

# Where we store our SQLite database — keeps track of plans we rejected
DB_PATH = "blog_planner.db"

def init_db():
    """Sets up the database so we can save plans that didn't make the cut."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create our table if it doesn't exist yet — first run safety net
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rejected_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        blog_title TEXT,
        sections TEXT,
        feedback TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def insert_rejected_plan(topic: str, blog_title: str, sections: List[str], feedback: str):
    """Saves a plan we rejected so we don't make the same mistake twice."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Store the plan along with what went wrong — we'll use this as memory later
    cur.execute("""
    INSERT INTO rejected_plans (topic, blog_title, sections, feedback)
    VALUES (?, ?, ?, ?)
    """, (topic, blog_title, "\n".join(sections), feedback))

    conn.commit()
    conn.close()


def fetch_recent_rejections(limit: int = 5) -> List[Dict]:
    """Grabs the last few rejected plans to help us learn from them."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    SELECT topic, blog_title, sections, feedback
    FROM rejected_plans
    ORDER BY created_at DESC
    LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    conn.close()

    # Turn raw DB rows into nice dictionaries we can actually work with
    return [
        {
            "topic": r[0],
            "blog_title": r[1],
            "sections": r[2],
            "feedback": r[3],
        }
        for r in rows
    ]


# Pydantic Models for our data structure
class Task(BaseModel):
    """Holds the details for one specific section of the blog."""
    id: int
    title: str
    goal: str = Field(..., description='What the reader should walk away with after reading this part.')
    bullets: List[str] = Field(..., min_length=3, max_length=6, description='3 to 6 key points we need to cover here, no fluff.')
    target_words: int = Field(..., description='How long this section should be (aim for 120-550 words).')
    section_type: Literal["intro", "core", "examples", "checklist", "common_mistakes", "conclusion"] = Field(
        ...,
        description="Type of section. Make sure 'common_mistakes' appears exactly once.",
    )
    tags: List[str] = Field(default_factory=list, description='Tags to help categorize this section.')
    requires_research: bool = Field(default=False, description='True if we need to look things up for this part.')
    requires_citations: bool = Field(default=False, description='True if we need to cite our sources.')
    requires_code: bool = Field(default=False, description='True if valid code examples are needed here.')


class RouterDecision(BaseModel):
    """What the router decided: do we need to research or not?"""
    needs_research: bool
    mode: Literal['open_book', 'closed_book', 'hybrid']
    queries: List[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """A single piece of useful info we found online."""
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class Plan(BaseModel):
    """The master plan for the entire blog post."""
    blog_title: str = Field(description="The catchy title of our blog.")
    audience: str = Field(description='Who are we writing for?')
    tone: str = Field(description='The vibe of the article (e.g., crisp, practical).')
    tasks: List[Task] = Field(description='The list of sections we need to write.')
    blog_kind: str = Field(default="tutorial", description='What kind of blog is this?')
    constraints: str = Field(default="", description='Any special rules we need to follow.')


class EvidencePack(BaseModel):
    """A bundle of all the evidence we gathered."""
    evidence: List[EvidenceItem] = Field(description="All the facts we found to back up our writing.")


class ImageSpec(BaseModel):
    """Details for an image we want to generate."""
    placeholder: str = Field(..., description="The marker we use in text, like [[IMAGE_1]].")
    filename: str = Field(..., description="Filename to save needed image, e.g., cool_chart.png.")
    alt: str
    caption: str
    prompt: str = Field(..., description="Instructions for the image generator.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    """The big picture plan for all images in the post."""
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class BlogWriter(TypedDict):
    """Keeps track of everything happening as we write the blog."""
    topic: str                    # what we're writing about
    mode: str                     # open_book, closed_book, or hybrid
    needs_research: bool          # do we need to search the web first?
    queries: List[str]            # search queries if research is needed
    evidence: List[EvidenceItem]  # facts we found from web research
    plan: Optional[Plan]          # the blog outline (None until we create it)
    approved: bool                # did the human give the thumbs up?
    sections: Annotated[List[tuple[int, str]], operator.add]  # written sections collected from workers
    merged_md: str                # all sections stitched together
    md_with_placeholders: str     # markdown with [[IMAGE_X]] placeholders
    image_specs: List[dict]       # specs for images we want to generate
    final: str                    # the finished blog post
    feedback: str                 # human feedback if they rejected the plan


# Our main LLM — Gemini 2.5 Flash does the heavy lifting for everything
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""


def router(state: BlogWriter) -> dict:
    """Decides if we need to hit the web for answers."""
    topic = state['topic']

    # Ask the LLM to classify the topic and decide our approach
    decider = llm.with_structured_output(RouterDecision)
    result = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f'topic: {topic}')
    ])

    return {
        "needs_research": result.needs_research,
        "mode": result.mode,
        "queries": result.queries,
    }


def route_next(state: BlogWriter) -> Literal['research', 'orchestrator']:
    """Points us to research or straight to planning, depending on what we need."""
    if state['needs_research']:
        return 'research'
    else:
        return 'orchestrator'


def tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """Uses Tavily to hunt down good web pages."""
    tool = TavilySearchResults(max_results=max_results)
    result = tool.invoke({'query': query})
    normalised: List[dict] = []

    # Tavily can return results in different formats, so we normalize everything
    if isinstance(result, list):
        for r in result:
            if isinstance(r, dict):
                normalised.append({
                    'title': r.get('title') or r.get('name') or '',
                    'url': r.get('url') or '',
                    'snippet': r.get('content') or r.get('snippet') or '',
                    'published_at': r.get('published_date') or r.get('published_at'),
                    'source': r.get('source'),
                })
    elif isinstance(result, str):
        # Sometimes Tavily just gives us a plain string — wrap it up nicely
        normalised.append({
            'title': query,
            'url': '',
            'snippet': result,
            'published_at': None,
            'source': 'tavily',
        })

    return normalised


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""


def research_node(state: BlogWriter) -> dict:
    """Gathers the research and cleans it up."""
    queries = state.get('queries', []) or []
    max_results = 6

    raw_results: List[dict] = []

    # Run each query through Tavily and collect everything
    for q in queries:
        raw_results.extend(tavily_search(q, max_results=max_results))

    # Nothing found? No worries, we'll just write without evidence
    if not raw_results:
        return {'evidence': []}

    # Let the LLM clean up and structure the raw search results
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results:\n{raw_results}")
    ])

    # Remove duplicates — if two results point to the same URL, keep just one
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {'evidence': list(dedup.values())}


ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""


def orchestrator(state: BlogWriter) -> dict:
    """Builds the master plan for the blog post."""
    planner = llm.with_structured_output(Plan)
    evidence = state.get('evidence', [])
    mode = state.get('mode', 'closed_book')

    # Pull up past rejections so the LLM doesn't repeat old mistakes
    recent_rejections = fetch_recent_rejections()
    memory_text = ""
    if recent_rejections:
        memory_text = "\n\n### Previously Rejected Plans (Avoid repeating):\n"
        for r in recent_rejections:
            memory_text += (
                f"- Topic: {r['topic']}\n"
                f"  Rejected title: {r['blog_title']}\n"
                f"  Rejected sections: {r['sections']}\n"
                f"  Human feedback: {r['feedback']}\n"
                f"  → Generate a DIFFERENT and IMPROVED structure.\n\n"
            )

    # Feed the system prompt + memory + evidence to the LLM and get a plan back
    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM + memory_text),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n"
                f"Mode: {mode}\n\n"
                f"Evidence (ONLY use for fresh claims; may be empty):\n"
                f"{[e.model_dump() for e in evidence][:16]}"
            )
        ),
    ])
    return {"plan": plan}


def hitl(state: BlogWriter) -> dict:
    """Shows the plan to a human (you!) for approval."""
    plan = state["plan"]
    assert plan is not None

    # Pretty-print the plan so the human can review it easily
    print("\n" + "="*60)
    print("HUMAN REVIEW - Blog Plan")
    print("="*60)
    print(f"\nTitle: {plan.blog_title}")
    print(f"Audience: {plan.audience}")
    print(f"Tone: {plan.tone}")
    print(f"\nSections ({len(plan.tasks)}):")
    print("-" * 60)
    for t in plan.tasks:
        print(f"\n{t.id}. {t.title}")
        print(f"   Goal: {t.goal}")
        print(f"   Type: {t.section_type} | Words: {t.target_words}")

    print("\n" + "="*60)
    decision = input("\nApprove this plan? (y/n): ").strip().lower()

    # Thumbs up? Let's go write that blog!
    if decision == "y":
        return {"approved": True}

    # Not happy with it? Tell us what to fix
    raw_feedback = input("\nWhat should be improved? ")

    # Use the LLM to clean up the feedback into something structured
    response = llm.invoke([
        SystemMessage(content="""You are a senior technical writer. Rewrite the user's feedback into clear, actionable improvement guidance in 4–8 concise lines. Return plain text only."""),
        HumanMessage(content=raw_feedback)
    ])

    refined_feedback = response.content.strip()

    # Save this rejection so the next attempt can learn from it
    insert_rejected_plan(
        topic=state["topic"],
        blog_title=plan.blog_title,
        sections=[t.title for t in plan.tasks],
        feedback=refined_feedback,
    )

    print(f"\n[Feedback recorded: {refined_feedback}]\n")

    return {
        "approved": False,
        "feedback": refined_feedback,
    }


def route_to_workers(state: BlogWriter):
    """If approved, sends work to the writers. If not, shuts it down."""
    if state['approved']:
        # Fan out — each section gets its own worker running in parallel, that's the magic!
        return [
            Send(
                "worker",
                {
                    "task": task.model_dump(),
                    "topic": state["topic"],
                    "mode": state["mode"],
                    "plan": state["plan"].model_dump(),
                    "evidence": [e.model_dump() for e in state.get("evidence", [])],
                },
            )
            for task in state["plan"].tasks
        ]
    else:
        # Plan got rejected, wrap things up
        return [Send("finish", state)]


def finish(state: BlogWriter) -> dict:
    """Simple cleanup if the plan was rejected."""
    print("\n[Plan rejected - workflow ending]")
    return {}


def route_after_hitl(state:BlogWriter):
    """Simple fork — approved plans go to fanout, rejected ones end here."""
    if state['approved']:
        return 'fanout'
    else :
        return END


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""


def worker_node(payload: dict) -> dict:
    """Writes one specific section of the blog post."""
    # Unpack everything from the payload into proper typed objects
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    # Format bullet points as a nice list for the LLM prompt
    bullets_text = "\n- " + "\n- ".join(task.bullets)

    # Format evidence URLs so the LLM knows what it can cite
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    # Ask the LLM to write the actual section — this is where the magic happens
    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    # Return with the section ID so we can sort them later
    return {"sections": [(task.id, section_md)]}


def merging_content(state: BlogWriter) -> dict:
    """Stitches all the written sections together into one document."""
    plan = state['plan']
    # Sort sections by their ID so they appear in the right order
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    # Slap the blog title on top and combine everything
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""


def decide_images(state: BlogWriter) -> dict:
    """Figures out if and where we need images."""
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Asks Gemini to draw a picture for us.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    # Grab the API key — can't do anything without it
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # The SDK response structure can vary, so we try multiple ways to get the image
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")
    
    # Dig through the response parts to find the actual image bytes
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def generate_and_place_images(state: BlogWriter) -> dict:
    """Creates the images and slots them into the final file."""
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # Sanitize filename
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in plan.blog_title)
    safe_title = safe_title.strip().replace(' ', '_')[:100]  # Limit length
    output_filename = f"{safe_title}.md"

    # If no images requested, just write merged markdown
    if not image_specs:
        Path(output_filename).write_text(md, encoding="utf-8")
        print(f"\n✓ Blog saved to: {output_filename}")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        img_filename = spec["filename"]
        out_path = images_dir / img_filename

        # Generate only if needed
        if not out_path.exists():
            try:
                print(f"Generating image: {img_filename}...")
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
                print(f"✓ Image saved: {out_path}")
            except Exception as e:
                print(f"✗ Image generation failed: {e}")
                # Graceful fallback
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{img_filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    Path(output_filename).write_text(md, encoding="utf-8")
    print(f"\n✓ Blog with images saved to: {output_filename}")
    return {"final": md}


# --- Reducer subgraph: merges sections → decides images → generates them ---
reducer_graph = StateGraph(BlogWriter)
reducer_graph.add_node("merging_content", merging_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merging_content")
reducer_graph.add_edge("merging_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()
reducer_subgraph


# --- Main workflow graph: the full pipeline from topic to finished blog ---
# Flow: router → (maybe research) → orchestrator → human review → workers → reducer → done!
graph = StateGraph(BlogWriter)

# Register all the nodes
graph.add_node('router', router)
graph.add_node('research', research_node)
graph.add_node('orchestrator', orchestrator)
graph.add_node('hitl', hitl)
graph.add_node('worker', worker_node)
graph.add_node('reducer', reducer_subgraph)
graph.add_node('finish', finish)

# Wire up the edges — this is how the workflow flows
graph.add_edge(START, 'router')
graph.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
graph.add_edge("research", "orchestrator")
graph.add_edge('orchestrator', 'hitl')
graph.add_conditional_edges('hitl', route_to_workers)
graph.add_edge('worker', 'reducer')
graph.add_edge('reducer', END)
graph.add_edge('finish', END)

# Compile and we're ready to roll
workflow = graph.compile()
workflow


def run(topic: str):
    """Kicks off the whole blog writing process."""
    print(f"\n{'='*60}")
    print(f"Starting Blog Writer for topic: {topic}")
    print(f"{'='*60}\n")
    
    # Start with a clean slate — every field gets its default value
    out = workflow.invoke({
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "approved": False,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "feedback": "",
    })
    
    print(f"\n{'='*60}")
    print("Workflow Complete!")
    print(f"{'='*60}\n")
    
    return out


if __name__ == "__main__":
    # Start up the database — gotta have this ready before anything else
    init_db()
