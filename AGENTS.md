# Agent Policy

This repository requires explicit human approval for all git publication actions.

- Agents must not run `git commit` unless a human explicitly asks for that commit in the current conversation.
- Agents must not run `git push` unless a human explicitly asks for that push in the current conversation.
- Agents must not create, update, or merge pull requests without explicit human approval.
- When in doubt, agents should stop and ask for confirmation before any publication action.
