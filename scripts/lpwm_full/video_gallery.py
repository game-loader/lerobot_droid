"""Build a local-only gallery of fixed first-episode videos from full-task eval JSONs."""

import argparse
import html
import json
from pathlib import Path


def build_gallery(root: Path) -> dict:
    """Keep failures visible; use only recorded results, and never auto-select success clips."""
    cards = []
    summary = {}
    metadata = []
    for scope in ("full130", "common40"):
        result_path = root / scope / "formal.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        if result["status"] != "complete":
            raise ValueError(f"Incomplete result: {scope}")
        summary[scope] = {
            "step": result["checkpoint"]["step"],
            "successes": result["successes"],
            "episodes": result["num_episodes"],
            "success_rate": result["success_rate"],
            "namespace": result["protocol"]["seed_namespace"],
            "videos": 0,
        }
        for task in result["per_task"]:
            for ep in task["episodes"]:
                if "video_error" in ep:
                    raise ValueError(f"Video failed: {scope} task{task['task_id']}: {ep['video_error']}")
                if "video" not in ep:
                    continue
                relative = Path(scope) / "formal_videos" / Path(ep["video"]).name
                if not (root / relative).is_file():
                    raise FileNotFoundError(root / relative)
                outcome = "success" if ep["success"] else "failure"
                label = "成功" if ep["success"] else "失败"
                name = html.escape(task["language"])
                suite = html.escape(task["suite"])
                src = html.escape(relative.as_posix(), quote=True)
                task_id = task["task_id"]
                task_rate = task["successes"] / len(task["episodes"])
                cards.append(
                    f'<article data-scope="{scope}" data-suite="{suite}" data-outcome="{outcome}">'
                    f"<header><b>{scope} · {suite} · task {task_id}</b>"
                    f'<span class="{outcome}">{label}</span></header>'
                    f'<video controls preload="none" playsinline src="{src}"></video>'
                    f"<h3>{name}</h3><p>固定 episode 0 · {ep['control_steps']} 控制步 · "
                    f"该任务完整评估 {task['successes']}/{len(task['episodes'])} ({task_rate:.0%})</p>"
                    f"<p>seed {ep['seed']} · init-state {ep['init_state_index']} · "
                    f'<a href="{src}" download>下载 MP4</a></p></article>'
                )
                summary[scope]["videos"] += 1
                metadata.append(
                    {
                        "scope": scope,
                        "suite": task["suite"],
                        "task_id": task_id,
                        "task_name": task["task_name"],
                        "language": task["language"],
                        "episode_index": ep["episode_index"],
                        "success": ep["success"],
                        "control_steps": ep["control_steps"],
                        "path": relative.as_posix(),
                    }
                )
    if not cards:
        raise ValueError("No completed video results available")
    stats = "".join(
        f'<div class="stat"><strong>{scope} · {d["step"]:,} step</strong><br>'
        f"<b>{d['success_rate']:.2%}</b> · {d['successes']}/{d['episodes']} episode<br>"
        f"{d['videos']} 个固定首回合视频</div>"
        for scope, d in summary.items()
    )
    overview_links = "".join(
        f'<p><a href="{scope}_overview.mp4">▶ {scope} 固定任务速览视频</a> · '
        f'<a href="{scope}_contact_sheet.jpg">关键帧预览</a></p>'
        for scope in summary
        if (root / f"{scope}_overview.mp4").is_file()
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LpWM · GB200 · best170k 评估视频</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#11151c;color:#e5eaf0}
body{max-width:1440px;margin:32px auto;padding:0 20px}h1{margin-bottom:8px}
p{color:#aeb9c8;line-height:1.5}.stats,.filters{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}
.stat{background:#1c2531;padding:18px 24px;border-radius:10px}.stat b{font-size:28px;color:#89c4ff}
select,input{padding:10px;border:1px solid #46556a;border-radius:6px;background:#1c2531;color:inherit}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:18px}
article{background:#1b2330;border:1px solid #2c394b;border-radius:10px;overflow:hidden}
article header{display:flex;justify-content:space-between;gap:8px;padding:12px;font-size:12px}
video{display:block;width:100%;aspect-ratio:1;background:#000;image-rendering:pixelated}
h3,article p{margin:12px;font-size:14px;overflow-wrap:anywhere}a{color:#89c4ff}
.success{color:#74e4a5}.failure{color:#ff9090}[hidden]{display:none!important}
</style><h1>LpWM · GB200 · 最佳 170k checkpoint</h1>
<p>2026-09-28 · 原生 LIBERO · 每任务 10 回合完整评估 · 8 task-level workers/GPU</p>
<p>每个任务固定展示第 1 回合，成功和失败均保留，并非挑选的成功演示。视频是模型实际看到的
128×128 agent-view 图像，20 fps、正常速度；网页放大不增加原始清晰度。
这次重放 validation 计划，不是独立 final/test。两个实验任务范围不同，且 LIBERO-10 seed 计划不完全匹配。</p>
<div class="stats">STATS</div><div>OVERVIEWS</div>
<div class="filters"><select id="scope"><option value="">两个实验</option><option>full130</option><option>common40</option></select>
<select id="suite"><option value="">全部套件</option><option>libero_spatial</option><option>libero_object</option><option>libero_goal</option><option>libero_90</option><option>libero_10</option></select>
<select id="outcome"><option value="">成功 + 失败</option><option value="success">成功</option><option value="failure">失败</option></select>
<input id="query" placeholder="搜索任务名称 / 指令"><span id="count"></span></div>
<div id="grid">CARDS</div>
<script>
const fields=['scope','suite','outcome','query'];
function filter(){let n=0;document.querySelectorAll('article').forEach(card=>{
 const show=fields.slice(0,3).every(k=>!document.getElementById(k).value||card.dataset[k]===document.getElementById(k).value)
 &&card.textContent.toLowerCase().includes(document.getElementById('query').value.toLowerCase());
 card.hidden=!show;if(show)n++;else card.querySelector('video').pause();});
 document.getElementById('count').textContent=n+' 个视频';}
fields.forEach(k=>document.getElementById(k).addEventListener('input',filter));filter();
document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>{
 document.querySelectorAll('video').forEach(other=>{if(other!==v)other.pause();});}));
</script></html>"""
    (root / "index.html").write_text(
        page.replace("STATS", stats).replace("OVERVIEWS", overview_links).replace("CARDS", "\n".join(cards))
    )
    (root / "video_index.json").write_text(json.dumps({"summary": summary, "videos": metadata}, indent=2))
    return {"summary": summary, "video_count": len(metadata), "gallery": str(root / "index.html")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_gallery(args.root), indent=2))


if __name__ == "__main__":
    main()
