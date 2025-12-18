- whenever acting on a prompt entails editing files log the full prompt
(if the prompt comes from e.g. a TODO comment, include eventual lines below the
TODO if they has to do with the prompt)
along with a list of files + additions/deletions
and a timestamp
as an append to the file './agents_log.md'
- python: prefer pathlib for path handling
