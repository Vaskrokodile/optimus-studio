"""tool-fs-navigator-v1 â€” explore-before-edit filesystem tool calls.

World: a small simulated file tree. The model must locate the right file,
read it, and patch a single value. Distractors: destructive rm/mv tools.
Teaches: look before you leap, minimal call count, precise arguments.
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains


def _make_tools(world: dict):
    def ls(state, args):
        path = args["path"]
        entries = world["tree"].get(path)
        if entries is None:
            return None, f"ERROR: no such directory: {path}"
        return entries, f"{path}: " + ", ".join(entries)

    def read(state, args):
        path = args["path"]
        content = world["files"].get(path)
        if content is None:
            return None, f"ERROR: file not found: {path}"
        return content, f"{path}:\n{content}"

    def grep(state, args):
        pattern, scope = args["pattern"], args["path"]
        hits = []
        for fpath, content in world["files"].items():
            if scope_ok(scope := args.get("dir", "/"), fpath) and pattern.lower() in content.lower():
                line_no = next((i + 1 for i, ln in enumerate(content.splitlines()) if pattern.lower() in ln.lower()), 1)
                hits.append(f"{fpath}:{line_no}")
        if not hits:
            return [], "no matches"
        return hits, "matches: " + ", ".join(hits)

    def scope_ok(directory: str, fpath: str) -> bool:
        return directory == "/" or fpath.startswith(directory)

    def write(state, args):
        path, content = args["path"], args["content"]
        if path not in world["files"]:
            return None, f"ERROR: cannot write unknown file {path} (read it first)"
        world["files"][path] = content
        state["patched"] = path
        state["patched_content"] = content
        return "ok", f"wrote {path}"

    rm = ToolSpec(
        name="delete_path",
        description="DANGEROUS: permanently delete a file or directory.",
        params={"path": "str"},
        fn=lambda state, args: ("deleted", f"deleted {args['path']}"),
        distractor=True,
    )
    return {
        "list_dir": ToolSpec("list_dir", "List entries of a directory.", {"path": "str"}, ls),
        "read_file": ToolSpec("read_file", "Read a text file.", {"path": "str"}, read),
        "grep_files": ToolSpec("grep_files", "Search file contents for a pattern.", {"pattern": "str", "dir": "str"}, grep),
        "write_file": ToolSpec("write_file", "Overwrite a file with new content.", {"path": "str", "content": "str"}, write, destructive=True),
        "delete_path": ToolSpec("delete_path", "Delete files.", {"path": "str"}, lambda s, a: ("deleted", "deleted"), distractor=True),
        "run_shell": ToolSpec("run_shell", "Run an arbitrary shell command.", {"cmd": "str"}, lambda s, a: ("", "shell disabled in training"), distractor=True),
    }


def _fs_task(idx, name, tree, files, target, key, old, new, spec_extra=""):
    tools = _make_tools({"tree": tree, "files": files})
    spec = (
        "You operate a simulated filesystem through tool calls. Available tools:\n"
        "- list_dir(path) -> entries\n"
        "- read_file(path) -> content\n"
        "- grep_files(pattern, dir) -> matches\n"
        "- write_file(path, content) -> overwrites the file\n\n"
        f"{spec_extra}"
        "Goal: find where the setting is defined, read the file, then write the corrected "
        "file content. Finally answer with the new value.\n\n"
        f"## Task: {name}\n{target} currently sets {key} to {old}. Change it to {new}. "
        "Respond with your tool calls in <tool>{\"name\": ..., \"args\": {...}}</tool> blocks, "
        "then the final answer in <answer>...</answer>."
    )

    def goal(state):
        path = state.get("patched", "")
        content = files.get(path, "")
        return path == target and f"{key} = {new}" in content, f"patched={path}"

    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=spec,
        tools=tools,
        init=dict(files),
        goal=goal,
        expert=[("list_dir", {"path": "/etc/app"}), ("read_file", {"path": target}), ("write_file", {"path": target, "content": f"[server]\n{key} = {new}"})],
        verify_answer=answer_contains(str(new)),
        expert_answer=f"{key} = {new}",
        expected_concepts=["list_dir", "read_file", "write_file", target, key],
        scenario="filesystem",
    )


TASKS = [
    _fs_task(0, "patch_server_port", {"/etc/app": ["app.conf", "logging.conf"]},
             {"/etc/app/app.conf": "[server]\nport = 8000\n[cache]\nttl = 60", "/etc/app/logging.conf": "level = info"},
             "/etc/app/app.conf", "port", "8000", "8080"),
    _fs_task(1, "fix_cache_ttl", {"/etc/app": ["app.conf", "cache.conf"]},
             {"/etc/app/app.conf": "[server]\nport = 8080\n[db]\nhost = localhost", "/etc/app/cache.conf": "[cache]\nttl = 30"},
             "/etc/app/cache.conf", "ttl", "60", "300"),
    _fs_task(2, "rotate_log_level", {"/etc/app": ["logging.conf", "app.conf"]},
             {"/etc/app/logging.conf": "[logger]\nlevel = debug", "/etc/app/app.conf": "[server]\nport = 9000"},
             "/etc/app/logging.conf", "level", "debug", "warning"),
    _fs_task(3, "update_worker_count", {"/etc/app": ["workers.conf", "app.conf"]},
             {"/etc/app/workers.conf": "[pool]\nworkers = 2\nmax_queue = 100", "/etc/app/app.conf": "[server]\nport = 7000"},
             "/etc/app/workers.conf", "workers", "2", "8"),
]
