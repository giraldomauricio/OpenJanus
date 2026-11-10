---
name: threat-model-diagrams
description: Use when the user wants an architecture, data-flow or threat-model diagram (draw.io / diagrams.net), wants to turn STRIDE threat-model results or database tables into a diagram, or wants to read, extend or modify an existing .drawio file.
tools: create_diagram, read_diagram, list_diagrams, draft_diagram_from_database, query, describe_table, find_relationships
---
# Creating draw.io diagrams

You never write draw.io XML. You build a structured description and call `create_diagram`; it lays everything out and saves a
`.drawio` file the user downloads from a link in the chat and opens in draw.io / diagrams.net.

## Workflow
1. **Gather the facts.** Sources can be the user's message, rows in the database (`describe_table`, `query`), or an existing
   file: call `list_diagrams`, then `read_diagram` and work from what it returns. Do not invent components or flows the
   source does not mention. If something essential is missing (what talks to what, where the trust boundaries are), ask.
2. **Model it.**
   - `nodes`: one per component. `kind`: `actor` (person / external user), `external` (third-party system), `process`
     (logic, ellipse in a DFD), `service` (app / API), `datastore` (database, bucket, file), `queue` (broker / stream).
   - `edges`: one per data flow, labelled with what flows and the protocol ("HTTPS: login"). Use `bidirectional` only when
     data truly flows both ways; `dashed` for asynchronous or out-of-band flows.
   - `groups`: zones. Set `boundary: true` for **trust boundaries** (internet / DMZ / internal / cloud account); each node
     belongs to at most one group. Groups cannot be nested: pick the most meaningful level.
   - Use short ids (`api`, `db`); labels are for humans.
3. **Create** with `create_diagram` (`direction` LR for flows that read left to right, TB for layered stacks). Keep diagrams
   readable: more than about 25 nodes is usually better split into several diagrams (overview + per area).
4. **Report**: say what you drew (counts, boundaries, anything you assumed) and give the download. The link appears in the chat
   under the tool result, so do not paste paths. Offer to adjust.

## From an image (screenshot, photo of a whiteboard, exported diagram)
The user attaches it with the 📎 button (or pastes it) and you receive it with their message. This is the least precise source,
so **never create the diagram straight away**:
1. Write down what you can see: components (with the kind you would give each), flows (from → to, label, direction), and zones /
   trust boundaries. Quote the labels as written; do not "fix" or translate names.
2. Mark everything you are unsure about (blurry text, an arrow whose end is unclear, an unlabeled box) and say what you assumed.
   Never invent components, flows or boundaries that are not visible. If the image has no readable architecture, say so.
3. Ask the user to confirm or correct the list. Only after they answer, call `create_diagram`.
4. If you cannot see the image (the app reports that the model cannot read images), say so and suggest Claude, Gemini or a
   vision-capable Ollama model; do not describe a diagram you have not seen.

## Guessing from the database
Only when the user asks you to work out the architecture from what is in the database, or when there is no other source:
1. Look at the tables (`list_tables`-style questions via `describe_table`/`query`). Two different situations:
   - **Tables that describe a system** (rows are components, services, assets, dependencies, flows, threats: e.g. `systems`,
     `connections`, `threats`): read the rows with `query` and build the diagram from them like any other source. This is
     real data about the architecture, so treat it as the user's statement of fact (still note odd or missing values).
   - **Ordinary business data** (customers, orders...): call `draft_diagram_from_database`. It only shows how tables relate.
2. Check `enough_data` in its answer. If false, **do not guess**: say what is missing and ask for a description or an image.
3. If true, review the `draft` and the evidence. Confirm `low` confidence relationships with the user. Then call
   `create_diagram`, and tell the user clearly that this is the inferred **data model**, that dashed lines are inferred links,
   and that services, networks and trust boundaries cannot be known from tables: offer to add them if they describe them.

## STRIDE threat models
When the user supplies threats (or asks you to analyse the architecture with STRIDE), attach them to the element they apply to,
as `threats` on a node or an edge:
- `category`: S Spoofing, T Tampering, R Repudiation, I Information disclosure, D Denial of service, E Elevation of privilege.
- `title`: the threat in one sentence; `severity`: low / medium / high (as stated by the user, otherwise your reasoned
  judgement; say that you chose it); `mitigation`: only if given or clearly warranted.
- Per STRIDE-per-element: external entities → S, R; processes → all six; data stores → T, R, I, D; data flows → T, I, D. Cross-boundary
  flows deserve the most scrutiny. Do not pad: only list threats that make sense for that element.
The diagram colours elements by their worst severity, shows the STRIDE letters on the shape, adds a legend, and puts the full
threat text (with mitigation) in the shape's tooltip in draw.io. If the user wants the threat list as a table, also give it in
your reply, grouped by element.

## Reading and changing an existing diagram
`read_diagram` returns the same structure `create_diagram` takes (kinds are guessed from shapes, so check them). To modify: read,
change the structure, call `create_diagram` with the same `name` to overwrite, or with a new name to keep the original.
Warn the user that the file is re-laid out, so manual positioning, colours and icons from the original are not preserved.
Pages: only one page is read at a time; the result lists the others (`page` selects).
