import os
from dashscope import MultiModalConversation
import argparse
import tqdm
from typing import Any, Optional
import json

PROMPT = '''
Please answer the following question about the given video. Answer with "yes" or "no".
Question: 
'''

def main(questions=None, video_path=None):

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Environment variable DASHSCOPE_API_KEY is not set")

    def extract_answer_text(resp: Any) -> Optional[str]:
        if resp is None:
            return None

        # Most common: dict response
        if isinstance(resp, dict):
            output = resp.get("output") or {}
            choices = output.get("choices") or []
            if not choices:
                return None
            msg = choices[0].get("message")
            if msg is None:
                return None
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        return str(part["text"])
                return str(content[0]) if content else None
            if content is not None:
                return str(content)
            return None

        # Fallback: object-like response
        try:
            output = getattr(resp, "output", None)
            choices = getattr(output, "choices", None) if output is not None else None
            if choices:
                msg = getattr(choices[0], "message", None)
                content = getattr(msg, "content", None) if msg is not None else None
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and "text" in first:
                        return str(first["text"])
                    return str(first)
            return None
        except Exception:
            return None
    score = 0
    res = []
    for i, q in enumerate(questions):
        prompt = PROMPT + q
        messages = [
            {
                'role': 'user',
                'content': [
                    {'video': video_path, 'fps': 8},
                    {'text': prompt}
                ]
            }
        ]
        
        response = MultiModalConversation.call(
            api_key=api_key,
            model='qwen3-vl-plus',
            messages=messages,
            temperature=0.0,
        )

        # Some failures return None; avoid crashing and surface debug info.
        if response is None:
            print("[WARN] DashScope returned None response")
            return None

        answer_text_raw = extract_answer_text(response)
        if not answer_text_raw:
            # Print minimal debug info without dumping huge payloads
            if isinstance(response, dict):
                print(
                    "[WARN] Could not extract answer text; keys:",
                    list(response.keys()),
                    "code:",
                    response.get("code"),
                    "message:",
                    response.get("message"),
                )
            else:
                print("[WARN] Could not extract answer text; response type:", type(response))
            return None

        answer_text = answer_text_raw.strip().lower()
        answer = None
        try:
            if 'yes' in answer_text:
                answer = 1
            elif 'no' in answer_text:
                answer = 0
        except:
            print(f"Error parsing answer: {answer_text}")
        score += answer
        res.append(answer)
    return score, res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, help='Path to the metadata JSON file containing video-question mappings')
    parser.add_argument('--video_folder', type=str, help='Path to the folder containing videos to evaluate')
    parser.add_argument('--results_path', type=str, help='Path to the output JSON file to save detailed results', default='distinctive_results_veo.json')
    args = parser.parse_args()

    with open(args.metadata, 'r') as f:
        data = json.load(f)

    score = 0
    success = 0
    detail = {}
    for k,v in tqdm.tqdm(data.items()):
        filename = f"{k}.mp4"
        questions = v.get("questions")
        ans, res = main(questions=questions, 
            video_path=os.path.join(args.video_folder, filename))
        if ans is None:
            print("[WARN] Got None answer; skipping from average. video:", filename)
            continue
        score += ans/len(res)
        success += 1

        detail[k] = {
            "total_score": ans,
            "per_question_score": res
        }
        
    detail['total_score'] = score/success
    detail['success'] = success
    with open(args.results_path, 'w') as f:
        json.dump(detail, f, indent=4)

    print(f"Final Score: {detail['total_score']}, over {success} videos successfully evaluated.")


