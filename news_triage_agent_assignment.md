# Assignment: Build a News Triage Agent

## Goal

Build an agentic system that takes a user query about a **current topic** and produces a **triage brief** grounded in recent sources.

The system should do more than summarize. It should:
- search for recent reporting,
- extract key entities and claims,
- classify sources and framing,
- assess the **veracity status** of the main claim,
- explain why different media or states may frame the story differently,
- add a short historical background,
- and return a structured final brief.

---

## Core output of the agent

Given a query such as:

> "Why are different newspapers describing the same event in opposite ways?"

the agent should return:

1. **Main claim(s)** found in recent reporting  
2. **Veracity assessment**
   - confirmed
   - partly confirmed
   - disputed
   - unsupported
   - unclear / not enough evidence
3. **Why outlets may frame it differently**
   - domestic political alignment
   - geopolitical interest
   - editorial ideology
   - economic dependence
   - fear of state pressure
   - audience incentives
4. **Why a country may stay silent**
   - strategic ambiguity
   - alliance management
   - internal vulnerability
   - censorship / repression risk
   - lack of verified information
5. **Brief historical context**
   - 3 to 6 lines maximum
   - only what is necessary to understand the present event
6. **Source profile**
   - country
   - medium type: newspaper / TV / state media / public broadcaster / private broadcaster / digital-only
   - rough political orientation if known: left / center-left / center / center-right / right / unclear
   - tone: neutral / analytical / partisan / sensational
7. **Final triage brief**
   - short and operational
   - what seems true,
   - what remains uncertain,
   - what narratives are competing

---

## Important rule: separate fact, inference, and speculation

The system must always distinguish between:

### A. Observation
What the source literally said.

Example:
> "Outlet X described the strike as a defensive operation."

### B. Evidence-backed inference
A reasoned interpretation supported by known facts.

Example:
> "This framing may reflect the outlet's pro-government editorial line and the country's alliance structure."

### C. Speculation
A hypothesis that is plausible but not well established.

Example:
> "The country's silence may be due to fear of retaliation, but the available evidence is insufficient to say this confidently."

The agent must **never present speculation as fact**.

---

## Why this project is a good fit

This assignment combines what you studied in:

### Transformer NLP
- token classification / NER
- sentiment / tone analysis
- intent classification
- summarization
- claim-focused text understanding

### Agentic AI
- sequential pattern
- coordinator / manager pattern
- loop pattern
- tool usage with search
- decision about when to stop and when to search again

So this is a good bridge between:
- **NVIDIA NLP topics**
- **Google/Kaggle Day 1B agent patterns**

---

## System design

### Agent architecture

#### 1. Coordinator / Manager agent
Responsibilities:
- understand the user query,
- decide the plan,
- route work to other agents,
- combine outputs,
- produce final answer.

#### 2. Search agent
Tool:
- web search

Responsibilities:
- search recent articles,
- gather diverse sources,
- avoid duplicates,
- collect a balanced initial set.

#### 3. Extraction and classification agent
Responsibilities:
- extract named entities,
- identify the main claim(s),
- classify source type,
- classify political leaning if possible,
- classify tone: neutral / analytical / partisan / sensational,
- detect the likely intent of the source text: inform / persuade / justify / attack / deflect.

#### 4. Context agent
Responsibilities:
- add short historical background,
- identify recurring narratives,
- explain why the same event may be framed differently across countries and outlets.

#### 5. Verification / loop agent
Responsibilities:
- decide whether evidence is sufficient,
- trigger another search if:
  - sources are too repetitive,
  - only one side is represented,
  - the main claim is still unclear,
  - the evidence is weak or contradictory.

---

## Recommended workflow pattern

### Sequential pattern
Use this first:
1. search
2. extract
3. classify
4. contextualize
5. summarize

### Coordinator pattern
Use the manager to:
- choose sub-agents,
- merge outputs,
- decide final structure.

### Loop pattern
Use a loop only when needed:
- if the claim is controversial,
- if framing is sharply divergent,
- if the evidence base is weak,
- if no credible confirmation is found.

Do **not** let the loop run too long.  
A good rule is:
- maximum 2 search rounds.

---

## Input examples

### Example 1
> "Why are French and American outlets describing this strike differently?"

### Example 2
> "Did the government really ban this group, or is this media exaggeration?"

### Example 3
> "Why are some Arab channels silent about this event?"

### Example 4
> "Compare how Moroccan, Algerian, French, and Spanish outlets framed this issue."

---

## Output schema

Use a structured output like this:

```json
{
  "query": "Why are French and American outlets describing this strike differently?",
  "main_claims": [
    {
      "claim": "Country A carried out a defensive strike.",
      "status": "partly confirmed",
      "evidence_level": "moderate"
    }
  ],
  "entities": {
    "countries": ["France", "United States", "Country A"],
    "people": [],
    "organizations": [],
    "locations": []
  },
  "source_profiles": [
    {
      "name": "Example Outlet",
      "country": "France",
      "type": "newspaper",
      "orientation": "center-left",
      "tone": "analytical"
    }
  ],
  "framing_analysis": [
    "French coverage emphasized diplomatic risk.",
    "American coverage emphasized security justification."
  ],
  "silence_analysis": [
    "Some regional outlets may be avoiding strong positioning because of alliance sensitivities."
  ],
  "historical_context": [
    "This issue has roots in earlier regional escalation.",
    "The same actors have used similar narratives in prior crises."
  ],
  "final_brief": "The core event appears real, but justification and responsibility are framed differently across outlets. Part of the divergence seems linked to domestic politics, alliance structures, and editorial positioning.",
  "uncertainties": [
    "The exact trigger remains contested."
  ],
  "loop_used": true
}
```

---

## NLP components to use

### 1. Named Entity Recognition (NER)
Purpose:
- extract countries,
- people,
- organizations,
- institutions,
- locations,
- dates,
- possibly laws or operations.

Use it to detect:
- who is involved,
- where the event happened,
- which actors are being emphasized or omitted.

### 2. Sentiment / tone classification
This should not be only positive / negative.

A better media-focused label set is:
- neutral
- analytical
- alarmist
- partisan
- accusatory
- sensational

### 3. Intent classification
Classify the user's request:
- summarize
- compare narratives
- verify a claim
- explain silence
- explain bias / framing
- add historical context

Optional: classify the article's rhetorical intent too:
- inform
- persuade
- justify
- delegitimize
- distract
- mobilize

### 4. Claim extraction
The system should identify the central factual or political claims in each source.

### 5. Brief summarization
Produce short source summaries before producing the final synthesis.

---

## Suggested technical stack

You can keep it simple.

### Option A: Hugging Face + custom orchestration
- NER model from Hugging Face
- text classifier for tone / topic / intent
- summarization model or LLM
- your own Python orchestration

### Option B: Google ADK style
- manager agent
- search agent with search tool
- extraction/classification agent
- loop agent for re-search

### Option C: hybrid
- use HF models for structured NLP tasks,
- use an LLM agent for reasoning and synthesis.

---

## Suggested patterns and why

### Pattern 1: Search -> Extract -> Judge
Use this when the query is mainly about verification.

### Pattern 2: Search -> Compare -> Explain framing
Use this when the query is about bias, media angle, or competing narratives.

### Pattern 3: Search -> Compare -> Add history -> Judge uncertainty
Use this when the topic is geopolitical or historically loaded.

### Pattern 4: Loop until evidence threshold is reached
Use this only when:
- contradiction is high,
- the claim is recent,
- the source set is too narrow.

---

## Practical constraints

To keep the project short:

- start with **3 to 5 sources** only,
- use **one search tool**,
- use **one NER model**,
- use **one classifier** for tone or topic,
- do not build multi-agent memory yet,
- limit the loop to **2 rounds**.

---

## Evaluation criteria

Judge the system on:

### 1. Retrieval quality
- Are the sources relevant?
- Are they diverse?
- Are they recent?

### 2. Extraction quality
- Did it identify the right entities?
- Did it identify the main claim?

### 3. Framing analysis quality
- Did it distinguish fact from narrative?
- Did it explain likely incentives carefully?

### 4. Veracity judgment quality
- Did it avoid false certainty?
- Did it separate evidence from speculation?

### 5. Final brief quality
- Is it concise?
- Is it structured?
- Is it useful?

---

## Deliverables

1. A notebook or script
2. A short README
3. A structured JSON output
4. A few tested example queries
5. A short reflection:
   - What worked?
   - What failed?
   - Where did the model hallucinate?
   - Where did source bias classification become uncertain?

---

## Example final brief

### Query
> "Why are some outlets calling this a defensive strike while others call it aggression?"

### Example answer
- The event itself appears broadly confirmed across multiple sources.
- The main divergence is not about whether something happened, but about **how it is morally and politically framed**.
- Sources aligned with governments or alliance systems tend to justify or soften the action.
- Critical or opposition-oriented outlets emphasize civilian impact, legality, or escalation risk.
- Some countries may remain silent because speaking clearly would create diplomatic cost or expose internal contradictions.
- Historically, this issue fits a longer pattern in which identical events are described differently depending on strategic alignment.

---

## Final note

The project should aim for **disciplined reasoning**, not theatrical certainty.

A strong version of this agent:
- searches well,
- extracts well,
- explains framing carefully,
- and admits uncertainty when needed.

That is better than an overconfident system that pretends to know hidden motives as facts.
