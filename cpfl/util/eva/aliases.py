from typing import Callable, Coroutine

from ipv8.types import Peer

from cpfl.util.eva.exceptions import TransferException
from cpfl.util.eva.result import TransferResult

TransferCompleteCallback = Callable[[TransferResult], Coroutine]
TransferErrorCallback = Callable[[Peer, TransferException], Coroutine]
TransferRequestCallback = Callable[[Peer, bytes], Coroutine]
