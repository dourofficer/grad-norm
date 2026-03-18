MAGENTIC_ONE = """
* Orchestrator: The lead agent, the Orchestrator is responsible for high-level \
planning, directing specialized agents, tracking progress, updating states and \
ledgers, and error recovery. As the Orchestrator operates, its planning can be \
generic or lack of specificity as long as it is not wrong.

* Assistant: This LLM-based agent is specialized in writing code, analyzing \
information collected by other agents, and creating new artifacts. It can author \
new programs and is capable of debugging its own code when provided with console \
output.

* ComputerTerminal: This agent is deterministic and does not use an LLM. Computer \
terminal performs no other action than running Python scripts (provided to it quoted \
in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code \
blocks). It merely does what was told.

* WebSurfer: This highly specialized LLM-based agent is proficient in managing a \
Chromium-based web browser. It receives natural-language requests and maps them to \
actions such as visiting URLs, performing web searches, clicking elements, typing \
into forms, and scrolling. It can also perform "reading actions" like summarizing \
content or answering questions about a document. For visual grounding, it uses \
"set-of-marks" prompting on annotated screenshots to interact with specific page \
elements. It can also be asked to sleep and wait for pages to load, in cases where \
the pages seem to be taking a while to load.

* FileSurfer: Similar in design to the WebSurfer, the FileSurfer commands a custom \
markdown-based file preview application instead of a web browser. This read-only \
application supports a wide range of file types, including PDFs, Office documents, \
images, videos, and audio. The FileSurfer can navigate folder structures and list \
directory contents to locate and process information within local files.
""".strip()