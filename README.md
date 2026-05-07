# animationbench

animationbench is a dimension-first implementation of the benchmark logic.
Each close-set or IP dimension is implemented in its own module so callers can
compose only the dimensions they need.

## Close-Set Dimensions

The close-set modules are under `close_set/`:

- `twelve_principles_anticipation.py`
- `twelve_principles_squash_and_stretch.py`
- `twelve_principles_follow_through_overlapping.py`
- `semantic_action.py`
- `semantic_color.py`
- `semantic_scene.py`
- `semantic_object.py`

The VLM-only modules expose `evaluate_questions(...)` and
`evaluate_video(...)`. `twelve_principles_squash_and_stretch.py` adds SAM-based
area analysis through `evaluate_video(...)`.

## IP Dimensions

The IP modules are under `ip/`:

- `appearance.py`
- `expression.py`
- `motion.py`

Each IP module exposes:

- `build_prompt(...)`
- `extract_questions(...)`
- `evaluate_video(...)`
- `generate_video(...)`

## Credentials

No credentials are stored in this package.

- Set `DASHSCOPE_API_KEY` for Qwen VLM evaluation.
- Set `POLLO_API_KEY` and `PUBLIC_DOMAIN` for Pollo video generation.

## Dependencies

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

SAM area analysis requires the SAM3, OpenCV, NumPy, Pillow, and Torch runtime
used by the original Squash and Stretch evaluator.

