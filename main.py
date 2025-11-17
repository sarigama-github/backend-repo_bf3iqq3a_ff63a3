import os
from typing import List, Literal, Optional, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# FastAPI app
app = FastAPI(title="Kinship Calculator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Kinship Calculator Backend Running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


class RelationshipRequest(BaseModel):
    steps: List[str]


def _normalize_token(tok: str) -> str:
    t = tok.strip().lower()
    aliases = {
        "mom": "mother",
        "mum": "mother",
        "dad": "father",
        "boy": "son",
        "girl": "daughter",
        "man": "husband",
        "woman": "wife",
        "bro": "brother",
        "sis": "sister",
        "parents": "parent",
        "kids": "child",
        "child": "child",
        "son": "son",
        "daughter": "daughter",
        "spouse": "spouse",
        "wife": "wife",
        "husband": "husband",
        "mother": "mother",
        "father": "father",
        "parent": "parent",
        "brother": "brother",
        "sister": "sister",
        "sibling": "sibling",
    }
    return aliases.get(t, t)


def _pluralize(base: str, n: int) -> str:
    return base if n == 1 else base + "s"


class RelationshipResult(BaseModel):
    label: str
    type: Literal["blood", "affinal", "none"]
    explanation: str
    canonical: str


@app.post("/api/relationship", response_model=RelationshipResult)
def compute_relationship(req: RelationshipRequest):
    steps = [_normalize_token(s) for s in req.steps]

    # Track counts
    up = 0  # generations up
    down = 0  # generations down
    lateral_sibling = 0  # number of sibling hops
    affinity = False  # any spouse edge encountered

    # Gender of final person when determinable
    final_gender: Optional[str] = None  # 'male' or 'female'

    # Process tokens into structural moves
    for i, tok in enumerate(steps):
        if tok in {"mother", "father", "parent"}:
            up += 1
            if tok == "mother":
                final_gender = "female"
            elif tok == "father":
                final_gender = "male"
            else:
                final_gender = None
        elif tok in {"son", "daughter", "child"}:
            down += 1
            if tok == "son":
                final_gender = "male"
            elif tok == "daughter":
                final_gender = "female"
            else:
                final_gender = None
        elif tok in {"brother", "sister", "sibling"}:
            # A sibling is up 1 then down 1 on a different branch
            up += 1
            down += 1
            lateral_sibling += 1
            if tok == "brother":
                final_gender = "male"
            elif tok == "sister":
                final_gender = "female"
            else:
                final_gender = None
        elif tok in {"spouse", "husband", "wife"}:
            affinity = True
            if tok == "husband":
                final_gender = "male"
            elif tok == "wife":
                final_gender = "female"
            else:
                final_gender = None
        else:
            # Unknown token; treat as neutral step that doesn't change structure
            pass

    # Helper to build labels with gender when applicable
    def gendered(masculine: str, feminine: str, neutral: str) -> str:
        if final_gender == "male":
            return masculine
        if final_gender == "female":
            return feminine
        return neutral

    canonical = {
        "up": up,
        "down": down,
        "lateral": lateral_sibling,
        "affinal": affinity,
    }

    # Direct spouse
    if steps and steps[-1] in {"spouse", "husband", "wife"} and len(steps) == 1:
        label = gendered("husband", "wife", "spouse")
        return RelationshipResult(
            label=label,
            type="affinal",
            explanation="Direct marital relation.",
            canonical=str(canonical),
        )

    # Only ups or only downs = ancestors/descendants
    if down == 0 and up > 0 and not affinity:
        if up == 1:
            label = gendered("father", "mother", "parent")
        elif up == 2:
            label = gendered("grandfather", "grandmother", "grandparent")
        else:
            greats = up - 2
            label = gendered(
                "great-" * greats + "grandfather",
                "great-" * greats + "grandmother",
                "great-" * greats + "grandparent",
            )
        return RelationshipResult(
            label=label,
            type="blood",
            explanation="Direct ancestor.",
            canonical=str(canonical),
        )

    if up == 0 and down > 0 and not affinity:
        if down == 1:
            label = gendered("son", "daughter", "child")
        elif down == 2:
            label = gendered("grandson", "granddaughter", "grandchild")
        else:
            greats = down - 2
            label = gendered(
                "great-" * greats + "grandson",
                "great-" * greats + "granddaughter",
                "great-" * greats + "grandchild",
            )
        return RelationshipResult(
            label=label,
            type="blood",
            explanation="Direct descendant.",
            canonical=str(canonical),
        )

    # Siblings, aunts/uncles, nieces/nephews
    if up == 1 and down == 1 and lateral_sibling >= 1 and not affinity:
        label = gendered("brother", "sister", "sibling")
        return RelationshipResult(
            label=label,
            type="blood",
            explanation="Child of your parent (not you).",
            canonical=str(canonical),
        )

    if up >= 2 and down == 1 and not affinity:
        # Parent's siblings -> aunts/uncles; with more ups => great-aunts/uncles
        gen = up - 1  # 1 => aunt/uncle; 2 => great-aunt/uncle, etc
        if gen == 1:
            base = gendered("uncle", "aunt", "aunt/uncle")
        else:
            base = gendered(
                "great-" * (gen - 1) + "granduncle",
                "great-" * (gen - 1) + "grandaunt",
                "great-" * (gen - 1) + "grand-aunt/uncle",
            )
        return RelationshipResult(
            label=base,
            type="blood",
            explanation="Sibling of an ancestor.",
            canonical=str(canonical),
        )

    if up == 1 and down >= 2 and not affinity:
        # Sibling's child(ren) -> nieces/nephews incl. great-
        gen = down - 1  # 1 => niece/nephew; 2 => grandniece/nephew (great-niece/nephew)
        if gen == 1:
            base = gendered("nephew", "niece", "niece/nephew")
        elif gen == 2:
            base = gendered("grandnephew", "grandniece", "grandniece/nephew")
        else:
            base = "great-" * (gen - 2) + gendered("grandnephew", "grandniece", "grandniece/nephew")
        return RelationshipResult(
            label=base,
            type="blood",
            explanation="Descendant of your sibling.",
            canonical=str(canonical),
        )

    # Cousins: go up g >= 2 to a common ancestor, then down h >= 2 on a different branch
    if up >= 2 and down >= 2 and not affinity:
        degree = min(up, down) - 1  # 1 => first cousin, 2 => second cousin, etc.
        removal = abs(up - down)
        # Build cousin label
        ordinals = {
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
        }
        cousin_core = f"{ordinals.get(degree, str(degree)+ 'th')} cousin"
        if removal == 0:
            label = cousin_core
        else:
            label = f"{cousin_core} {removal} {_pluralize('time', removal)} removed"
        return RelationshipResult(
            label=label,
            type="blood",
            explanation="Descended from a shared ancestor on a different branch.",
            canonical=str(canonical),
        )

    # Affinal relations via spouse connections
    if affinity:
        # Common patterns
        # sibling -> spouse => brother/sister-in-law
        if steps and steps[-1] in {"brother", "sister", "sibling"}:
            label = gendered("brother-in-law", "sister-in-law", "sibling-in-law")
            return RelationshipResult(
                label=label,
                type="affinal",
                explanation="Sibling of your spouse OR spouse of your sibling.",
                canonical=str(canonical),
            )
        # parent of spouse
        if up >= 1 and down == 0:
            if up == 1:
                label = gendered("father-in-law", "mother-in-law", "parent-in-law")
            elif up == 2:
                label = gendered("grandfather-in-law", "grandmother-in-law", "grandparent-in-law")
            else:
                greats = up - 2
                label = gendered(
                    "great-" * greats + "grandfather-in-law",
                    "great-" * greats + "grandmother-in-law",
                    "great-" * greats + "grandparent-in-law",
                )
            return RelationshipResult(
                label=label,
                type="affinal",
                explanation="Ancestor of your spouse.",
                canonical=str(canonical),
            )
        # child of spouse (step-children when not your own)
        if up == 0 and down >= 1:
            if down == 1:
                label = gendered("stepson", "stepdaughter", "stepchild")
            else:
                # Step-grandchildren
                if down == 2:
                    label = gendered("step-grandson", "step-granddaughter", "step-grandchild")
                else:
                    greats = down - 2
                    label = gendered(
                        "step-" + "great-" * greats + "grandson",
                        "step-" + "great-" * greats + "granddaughter",
                        "step-" + "great-" * greats + "grandchild",
                    )
            return RelationshipResult(
                label=label,
                type="affinal",
                explanation="Descendant of your spouse (not biologically yours).",
                canonical=str(canonical),
            )
        # Brother -> his wife -> her brother pattern: no direct relation
        return RelationshipResult(
            label="no direct relation",
            type="none",
            explanation="Related only by marriage through multiple links (e.g., your brother's wife's brother).",
            canonical=str(canonical),
        )

    # Fallbacks
    if up == 1 and down == 1 and not affinity:
        # This could be sibling or your own child then up; ambiguous
        return RelationshipResult(
            label="sibling (or self via parent)",
            type="blood",
            explanation="Ambiguous path interpreted as sibling.",
            canonical=str(canonical),
        )

    return RelationshipResult(
        label="unknown / ambiguous",
        type="none",
        explanation="The selected path doesn't map to a common English kinship term.",
        canonical=str(canonical),
    )


@app.get("/test")
def test_database():
    """Retained environment/database diagnostic endpoint"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Used",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    # Check environment variables presence only
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
