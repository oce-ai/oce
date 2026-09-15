"""oce bench —— 检索质量评测客户端。

与被测服务同仓、但通过 HTTP 对话的独立评测工具：跑查询集、算 Top-1 + nDCG@10、
产出结构化 run 记录与报告。不参与 DDD 分层（它是服务的**外部客户端**），只复用两个
纯函数真源：``application.service.compute_blob_name``（内容寻址）与
``domain.services.source_filter``（本地遍历与服务端索引用同一套准入规则）。

模块职责：
- ``scoring``   评分口径纯函数（Top-1 / nDCG@10 / glob 路径匹配）
- ``blobs``     本地仓库遍历 + 上传批次切分
- ``client``    HTTP 传输层（上传 / 等嵌入 / 检索 / 热改 read-after-write）
- ``harness``   编排层（一次评测 = 索引 + 跑查询 + 测资源）
- ``report``    markdown 渲染
- ``runrecord`` 结构化 run 记录（Commit 7）
- ``profiles``  profile 文件驱动的后端配置（Commit 5）
- ``sweep`` / ``compare`` / ``service`` / ``cli``  多组参数扫描与对比（Commit 6）
"""
