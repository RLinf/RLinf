# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Dict, Any, List


def make_info_json(
    codebase_version: str,
    robot_type: str,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    total_chunks: int,
    chunks_size: int,
    fps: int,
    splits: Dict[str, str],
    data_path: str,
    video_path: str,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": splits,
        "data_path": data_path,
        "video_path": video_path,
        "features": features,
    }


def make_episode_row(episode_index: int, tasks: List[str], length: int) -> Dict[str, Any]:
    return {"episode_index": episode_index, "tasks": tasks, "length": length}


def make_task_row(task_index: int, task: str) -> Dict[str, Any]:
    return {"task_index": task_index, "task": task}


def make_episode_stats_row(episode_index: int, stats: Dict[str, Any]) -> Dict[str, Any]:
    return {"episode_index": episode_index, "stats": stats}


