# multithread-iterator-chainer
A Multithreading dataset Iterator for Chainer

multi-threading iteratorを作りました。chainerのmultiprocess iteratorと同じ使い方で、名前を変えるだけで使えます。

multiprocess iteratorの以下の問題を解決します：
* main process killで殺すと後に変なプロセスが１０個くらい残る、いちいち殺すのも面倒。
* 場合によってはkillで殺せない。
* nvidia-smiでは見えないけど、なぜかGPUメモリをそれらのプロセスが握ってるらしい

走っているプロセスは一個だけ、killで殺してもトラブルなし。
速度も同じくらいのはずです。Python3が必要です。レビューされてないので無保証です。

---

This is a multi-threading dataset iterator for Chainer.  It can be used as a drop-in replacement for multiprocess iterator of Chainer.

The following nuisance with multiprocess iterator will be solved:
* If you kill the main process, 10 or more of subprocesses can remain, which is bothersome.
* Sometimes that worker subprocesses can't be killed.
* Sometimes the subprocesses can take GPU memory that can't be watched by nvidia-smi.

The number of the running process will be one, and no trouble in killing it.
The performance would be about the same as the multiprocess iterator.  
Python 3 is required.  No warrany.
