# Core concept

Synthetic REST API Test Data Generator

Input

OpenAPI spec (initially)

Output

realistic JSON payloads for testing the API

Purpose

generate request bodies
generate response examples
generate edge cases

_This solves a real problem. Writing mock payloads for APIs is tedious and repetitive._

## Simple architecture (good for a notebook)

Keep it minimal:

openapi_generator.ipynb

1 load OpenAPI spec
2 extract schemas
3 prompt model
4 generate JSON examples
5 validate output
6 compare models

## Workflow

Step 1 — parse OpenAPI

Extract:

- endpoints
- request schemas
- response schemas

Example schema:

``` json
{
  "User": {
    "id": "integer",
    "name": "string",
    "email": "string",
    "created_at": "datetime"
  }
}

```

Step 2 — prompt template

Example prompt:

Generate 5 realistic JSON objects matching this schema.

Constraints:

- valid JSON
- realistic values
- diverse examples

Schema:
{id:int, name:string, email:string, created_at:datetime}

Step 3 — run different models

Using HuggingFace pipeline:

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    device="cuda"
)

result = generator(prompt, max_new_tokens=200)

Then repeat with:

phi
llama
qwen
quantized versions

## What you measure

This is the interesting part for the assignment.

Compare:

JSON validity

Does the output parse?

json.loads(output)
schema compliance

Does it match types?

Example:

email → string
id → integer
diversity

Are all records identical?

hallucinated fields

Models love to invent fields.

Example:

"username"
"phone_number"

when schema didn’t include them.

## A nice stretch goal (but optional)

Generate edge-case test data:

Prompt variant:

Generate edge-case JSON payloads for testing.

Include:

- null values
- extremely long strings
- boundary numbers
- unusual unicode characters

This becomes extremely useful for API testing.

## Your evaluation table could look like this

| model | quantized | valid JSON | schema accuracy | hallucinations |
| ------ | --------- | ---------- | --------------- | -------------- |
| phi-mini | yes | 90% | medium | few |
| qwen-coder | yes | 98% | high | low |
| llama3 | yes | 85% | medium | some |

That’s a very clean assignment report.

---

## Why this idea is particularly good

It shows:

- LLM prompting

- structured generation

- evaluation

- model comparison

- quantization effects

And it's directly relevant to software engineering.
Interviewers love things like this.

## One small improvement I would suggest

Instead of just:

OpenAPI → examples

frame it as:

OpenAPI → Synthetic Test Suite

Meaning it generates:

valid payloads
invalid payloads
edge cases

That sounds much more interesting.

## Scope control suggestion

Limit the first version to:

Input:
OpenAPI schema

Output:
5 JSON examples per endpoint

That's enough for the assignment.

## One funny thing you will probably observe

Coder models often outperform general models here.

For example:

qwen-coder tends to produce cleaner JSON

phi sometimes drifts into explanation text

llama sometimes adds comments

Which is exactly the kind of insight the course wants you to discover.
