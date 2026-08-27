"""
ArtPaintingJava — RL environment for teaching a model to paint beautiful art in Java.

Environment concept:
  The model is given an art prompt (a style/subject/palette description) and
  must produce Java code using java.awt.Graphics2D that paints a beautiful
  image on a BufferedImage. The code is compiled and executed in-memory by
  JavaArtServer, the resulting image is scored by a CLIP aesthetic predictor,
  and the aesthetic score becomes the RL reward.

This trains the model to:
  1. Write syntactically correct Java (compilation is a hard gate)
  2. Use Graphics2D effectively (gradients, shapes, transforms, alpha blending)
  3. Produce visually appealing compositions (maximize aesthetic score)
  4. Explore diverse styles (prompt variety prevents mode collapse)

Reward design:
  - Compile gate: if the code doesn't compile, reward = 0
  - Execution gate: if paint() throws or times out, reward = 0
  - Aesthetic score: CLIP-based [0, 1] — the primary signal
  - Format bonus: small bonus for clean code structure
  - Diversity bonus: penalize producing the same image as previous samples
    in the batch (prevents mode collapse to one "trick" image)
  - Anti-pattern penalties: overly simple code (single fillRect), repetitive
    code, etc.
"""

from __future__ import annotations

import random
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Art prompt generation
# ---------------------------------------------------------------------------

# Style descriptors — these guide the model's creative direction
STYLES = [
    "abstract expressionism", "geometric abstraction", "impressionist",
    "art deco", "bauhaus", "minimalism", "pointillism", "cubism",
    "surrealism", "color field", "op art", "fauvism",
    "ukiyo-e inspired", "art nouveau", "constructivism", "suprematism",
    "de stijl", "futurism", "vorticism", "orphism",
    "liquid gradients", "fractal-like patterns", "cellular automata art",
    "voronoi mosaic", "flow field painting", "recursive tessellation",
    "caustic light patterns", "interference patterns", "moiré patterns",
    "perlin noise landscape", "wave interference art",
]

# Subject/element hints — what to paint
SUBJECTS = [
    "a serene landscape at golden hour", "a cosmic nebula with stars",
    "flowing organic forms", "intersecting geometric planes",
    "a field of flowers seen from above", "ripples on water",
    "a mountain range in fog", "a city skyline at dusk",
    "concentric energy waves", "a spiral galaxy",
    "crystalline structures", "a forest canopy in autumn",
    "ocean waves crashing", "a desert dune seascape",
    "northern lights over mountains", "a garden of light",
    "musical visualization", "a storm of particles",
    "layered translucent veils", "a sunrise through clouds",
    "geometric city", "a temple in the clouds",
    "abstract portrait of motion", "a river delta from above",
    "a coral reef pattern", "butterfly wing microstructure",
    "a stained glass window", "reflections on a lake",
    "a constellation map", "autumn leaves swirling",
]

# Color palette hints
PALETTES = [
    "warm sunset tones (orange, red, gold, deep purple)",
    "cool ocean blues and teals with white foam",
    "monochromatic grayscale with a single accent color",
    "vibrant complementary colors (blue-orange, red-green)",
    "earthy naturals (ochre, sienna, olive, umber)",
    "neon on black (electric blue, magenta, cyan)",
    "pastel spring (pink, lavender, mint, pale yellow)",
    "jewel tones (emerald, sapphire, ruby, amethyst)",
    "muted winter (slate blue, grey, frost white, muted green)",
    "fire and ice (deep red, bright orange, ice blue, white)",
    "golden hour (amber, peach, rose, lavender)",
    "tropical (turquoise, coral, lime, sunshine yellow)",
    "nocturnal (deep blue, indigo, violet, silver)",
    "autumn (burnt orange, rust, mustard, deep brown)",
    "high contrast black and white with grey gradients",
]

# Composition hints
COMPOSITIONS = [
    "centered radial composition", "rule of thirds with focal point",
    "asymmetric balance", "golden spiral composition",
    "all-over pattern with no single focal point",
    "diagonal flow from corner to corner", "framed border with central scene",
    "layered depth from foreground to background", "symmetrical mirror composition",
    "spiral vortex composition", "grid-based modular composition",
    "organic flowing curves", "stacked horizontal bands",
]


def art_painting_generator(seed: int) -> Problem:
    """Generate a random art painting problem."""
    rng = random.Random(seed)
    style = rng.choice(STYLES)
    subject = rng.choice(SUBJECTS)
    palette = rng.choice(PALETTES)
    composition = rng.choice(COMPOSITIONS)

    # Difficulty: more constraints = harder
    num_constraints = rng.randint(1, 3)
    constraints = rng.sample(
        [style, subject, palette, composition], min(num_constraints, 4)
    )

    prompt = _build_prompt(constraints, seed)
    difficulty = 0.3 + 0.15 * len(constraints)

    return Problem(
        id=f"art_{seed}",
        prompt=prompt,
        difficulty=min(difficulty, 0.9),
        token_budget=2048,
        source="art_painting_generator",
        metadata={
            "style": style,
            "subject": subject,
            "palette": palette,
            "composition": composition,
            "constraints": constraints,
            "seed": seed,
        },
    )


def _build_prompt(constraints: list[str], seed: int) -> str:
    """Build the prompt shown to the model."""
    constraint_text = "\n".join(f"  - {c}" for c in constraints)

    return textwrap.dedent(f"""\
        Write Java code that paints a beautiful artwork on a BufferedImage.

        Art direction:
        {constraint_text}

        Requirements:
        1. Define a public class with this exact method signature:
           public static BufferedImage paint(int width, int height)
        2. Use java.awt.Graphics2D and java.awt.image.BufferedImage.
        3. The canvas is {512}x{512} pixels. Use the width/height parameters.
        4. Create a visually striking, aesthetically beautiful image.
        5. Use techniques like: gradients, alpha blending, transforms,
           shapes, curves, noise, recursion, or any Graphics2D features.
        6. Do NOT read or write files. Do NOT use network. Just paint.
        7. Return the BufferedImage from paint().

        Write ONLY the Java source code in a ```java code block.
        The class name can be anything (e.g., Art, Painting, Canvas).
        Make it beautiful.""")


# ---------------------------------------------------------------------------
# Java code extraction
# ---------------------------------------------------------------------------

def extract_java_code(response: str) -> Optional[str]:
    """
    Extract Java source code from the model's response.

    Looks for ```java ... ``` blocks, or falls back to finding
    a class definition if no code block is present.
    """
    # Try fenced code block first
    pattern = r"```(?:java|Java)?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # Take the longest match (most likely the full program)
        return max(matches, key=len).strip()

    # Fallback: find from first 'import' or 'class' to last '}'
    class_match = re.search(r"(?:import\s+[\w.*]+\s*;\s*)*public\s+class\s+\w+", response)
    if class_match:
        start = class_match.start()
        # Find the matching closing brace
        depth = 0
        for i in range(class_match.end(), len(response)):
            if response[i] == '{':
                depth += 1
            elif response[i] == '}':
                depth -= 1
                if depth == 0:
                    return response[start:i + 1].strip()

    return None


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class ArtPaintingVerifier(Verifier):
    """
    Verifier for art painting Java code.

    This is a "soft" verifier — instead of checking correctness, it:
    1. Extracts Java code from the response
    2. Compiles and executes it via JavaExecutor
    3. Scores the resulting image via AestheticScorer
    4. Returns the aesthetic score as the verifier score

    The actual Java execution and aesthetic scoring are done outside the
    verifier (in the training loop) because they require GPU and JVM
    resources that shouldn't be embedded in the Gymnasium env. The verifier
    here just checks code format and extractability.
    """

    def __init__(self, problem: Problem):
        self._problem = problem

    def verify(self, response: str) -> VerifierResult:
        # Check if Java code can be extracted
        code = extract_java_code(response)
        if code is None:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics={"error": "no_java_code_found"},
            )

        # Check for required method signature
        if "paint" not in code:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics={"error": "no_paint_method"},
            )

        # Check for BufferedImage usage
        if "BufferedImage" not in code:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics={"error": "no_bufferedimage"},
            )

        # Format is valid — actual scoring happens in the training loop
        return VerifierResult(
            correct=True,
            score=0.5,  # placeholder; real score comes from aesthetic evaluation
            diagnostics={
                "code_extracted": True,
                "code_length": len(code),
            },
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ArtPaintingJavaEnv(BaseReasoningEnv):
    """
    Gymnasium environment for art painting in Java.

    Single-turn: the model gets an art prompt, produces Java code, and
    receives a reward based on the aesthetic quality of the rendered image.

    The actual image rendering and aesthetic scoring are performed by the
    training loop (which has access to the JavaExecutorPool and
    AestheticScorer), not by the environment's step() method. The
    environment handles prompt generation, code extraction, and format
    checking. The training loop calls compute_art_reward() to get the
    final reward.
    """

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator: Optional[Any] = None,
        reward_config: Optional[Any] = None,
        anti_pattern_detector: Optional[Any] = None,
        render_mode: Optional[str] = None,
    ):
        if problems is None and problem_generator is None:
            problem_generator = art_painting_generator
        super().__init__(problems, problem_generator, reward_config,
                         anti_pattern_detector, render_mode)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ArtPaintingVerifier(problem)

    def _check_format(self, response: str) -> float:
        """Check if the response has a clean Java code block."""
        # Bonus for having a fenced code block
        has_fence = bool(re.search(r"```(?:java|Java)?\s*\n.*?```", response, re.DOTALL))
        # Bonus for having the correct method signature
        has_signature = "BufferedImage paint" in response or "BufferedImage  paint" in response
        # Penalty for excessive explanation text (we want code, not essays)
        code = extract_java_code(response)
        if code is None:
            return 0.0
        code_ratio = len(code) / max(len(response), 1)

        bonus = 0.0
        if has_fence:
            bonus += 0.3
        if has_signature:
            bonus += 0.3
        if code_ratio > 0.7:
            bonus += 0.2  # mostly code, little prose
        elif code_ratio > 0.5:
            bonus += 0.1

        return min(bonus, 0.8)


# ---------------------------------------------------------------------------
# Reward computation (called by the training loop, not by env.step())
# ---------------------------------------------------------------------------

def compute_art_reward(
    response: str,
    aesthetic_score: float,
    compile_success: bool,
    execution_success: bool,
    code: Optional[str] = None,
    format_bonus: float = 0.0,
    diversity_penalty: float = 0.0,
) -> tuple[float, dict]:
    """
    Compute the final RL reward for an art painting response.

    This is called by the training loop after it has:
    1. Extracted Java code from the response
    2. Compiled and executed it via JavaExecutorPool
    3. Scored the resulting image via AestheticScorer

    Args:
        response: The model's full text response.
        aesthetic_score: CLIP aesthetic score in [0, 1].
        compile_success: Whether the Java code compiled.
        execution_success: Whether paint() ran without error.
        code: The extracted Java code (for complexity checks).
        format_bonus: Format bonus from the environment.
        diversity_penalty: Penalty for producing similar images (0-0.3).

    Returns:
        (reward, info_dict)
    """
    info = {
        "aesthetic_score": aesthetic_score,
        "compile_success": compile_success,
        "execution_success": execution_success,
        "format_bonus": format_bonus,
        "diversity_penalty": diversity_penalty,
    }

    # Hard gates: no compile or no execution = zero reward
    if not compile_success or not execution_success:
        info["reward_components"] = "gated"
        return 0.0, info

    # Base reward: aesthetic score (primary signal)
    reward = aesthetic_score

    # Format bonus: small additive
    reward += format_bonus * 0.1

    # Code complexity bonus: reward using advanced Graphics2D features
    if code:
        complexity_bonus = 0.0
        advanced_features = [
            "GradientPaint", "RadialGradientPaint", "AlphaComposite",
            "AffineTransform", "setTransform", "rotate", "scale",
            "CubicCurve2D", "QuadCurve2D", "GeneralPath", "Path2D",
            "setPaint", "TexturePaint", "BasicStroke",
        ]
        for feature in advanced_features:
            if feature in code:
                complexity_bonus += 0.01
        complexity_bonus = min(complexity_bonus, 0.1)
        reward += complexity_bonus
        info["complexity_bonus"] = complexity_bonus

    # Diversity penalty: prevent mode collapse
    reward -= diversity_penalty

    # Clamp to [0, 1.2] (can slightly exceed 1.0 with bonuses)
    reward = max(0.0, min(reward, 1.2))

    info["reward_components"] = "full"
    info["total_reward"] = reward
    return reward, info
