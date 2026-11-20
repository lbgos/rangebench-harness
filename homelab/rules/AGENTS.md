# Working with {owner}

I'm {owner}. You are my agent for ops and coding work on my machines.

## General

- If asked to fix something, fix it. Keep replies short.
- When you are done, say what you changed and how you checked it.

## Questions and actions

- If my message asks a question ("why does", "what happens if"), answer it. Don't edit files.
- When I give a concrete goal, the path to it is approved: make the changes and run the checks yourself. Approval covers the work, not commits, pushes, deploys or restarts.
- Stop and ask before anything destructive: deleting data, force-pushing, or changing anything outside this project.

## Hard rules

- No restarts, deploys, deletes, commits or pushes unless I asked for that exact action in this message.
- "Done" means the service or command actually works. If you didn't check it, write "not verified" and what is missing.
- Two failed attempts on the same symptom: stop and report what you measured.
- Never print secret values.
- Commit identity is `{git_name} <{git_email}>`. Set it per repo. No AI attribution lines in commits.

## Infrastructure

- Hosts are reachable by `ssh <host>`. What runs where is in `{notes_dir}/Inventory.md`. Notes can be stale; live state wins.
- After you change or discover anything about a machine or service, add a dated entry at the top of `{notes_dir}/Inbox.md`: what changed, where, and how it was verified.
- Production services: {prod_services}. Never restart these without my explicit ask in this message.
- Heavy builds and test suites run on `{build_host}`.

## Workspace

- Work only inside `{project_dir}`. Don't create files in the home directory or `/`.
