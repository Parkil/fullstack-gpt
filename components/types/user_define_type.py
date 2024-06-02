from typing import Sequence, Any

from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable

from typing import TypeAlias

BindingLLMType: TypeAlias = Runnable[PromptValue
                                     | str | Sequence[BaseMessage | list[str] | tuple[str, str]
                                                      | str | dict[str, Any]], BaseMessage]
