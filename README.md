# WebShop HINT-SD — Qwen2.5-3B, 4×A100 80GB

独立的 **Qwen2.5-3B-Instruct＋原版HINT-SD-Multi** 环境适配仓库。
使用 [HINT-SD官方源码](https://github.com/wgcyeo/HINT-SD)，固定提交
bc95b44192b06328a88773dbf3b345fa1c499093。上游loss、选中步骤mask、反向传播、
优化器和EMA实现不修改。不是GRPO＋HINT-SD混合，也不是我们的RETHINK方法；
没有Route-SFT、静态skill、外接Analyzer或付费API。

此仓库只运行3B；1.5B另有独立仓库。每个仓库各用四卡，不能把两个尺寸挤进
本配方的同四张卡。所有源码、模型和环境资源均来自公开固定来源，
不依赖作者电脑、私有服务器或私有数据。

## 1. 配方和四卡分工

| 项目 | 设置 |
| --- | --- |
| 初始化 | Qwen/Qwen2.5-3B-Instruct，固定revision |
| 方法 | 原版HINT-SD-Multi：失败回看、最多3步局部反馈、选中completion reverse KL |
| Teacher | Student LoRA的EMA，decay=.999，不是固定基模 |
| 参数 | 原版LoRA q_proj/v_proj，rank32、alpha64、dropout.05 |
| 优化 | LR5e-6，linear＋5% warmup；原版top100＋剩余概率桶，distillation microbatch1 |
| 每步采样 | batch1×accumulation2×每任务4条rollout，通常8 episodes/update |
| 预算 | 150 optimizer updates；初始、每5步及最终150固定140题验证 |
| WebShop | 原生1000商品synthetic goals，history2、最多15步、prompt4096左截断、response512 |
| 解码 | train/feedback温度.6；验证温度.4，一任务一次Student自由rollout |
| 保存 | 每20步及最终150；最近两份完整可续训断点＋最终Student adapter |

原版共置入口只支持单训练进程，不能直接改四进程DDP而声称全局batch没变。
每个独立仓库的四卡分工为：

| 默认物理GPU，可在.env修改 | 工作 |
| --- | --- |
| 0 | 单训练进程：Student＋EMA Teacher的LoRA训练/评分 |
| 1、2、3 | 三个同步vLLM TP1副本，执行rollout和EMA hindsight生成 |

请求分发给三个副本后恢复原顺序；公共vLLM LoRA API用全新版本ID同步当前
Student/Teacher。没有异步旧策略、扩大batch、四个梯度rank或量化。
设备布局和推理批次变化不保证与作者单卡运行逐位相同。

**HINT-SD的8 episodes/update不等于GRPO/SDAR/RLSD的128 episodes/update。**
比较150步时请同时报告episode/token/GPU时间。环境长度、验证及150步是本次
适配预算，不冒充原论文BFCL/AppWorld设置或成绩。

## 2. 原版方法怎样接WebShop

Student只读任务、当前页面和原生history，自由生成并执行动作。只有失败轨迹
交给EMA Teacher回看，生成最多3步短反馈。Teacher在同一实际Student前缀后
额外看到反馈，对原采样completion评分；Student只在这些选中completion做反向KL：

\[
\mathcal L=\operatorname{mean}_{\text{selected tokens}}
D_{\mathrm{KL}}\left(\pi_\theta\;\middle\|\;\operatorname{sg}\pi_{\bar\theta}\right).
\]

选中completion包含原think/action文本，不是只训练action标签内部。
成功轨迹、其他未选中turn、观察和历史不做此目标。无效JSON或未找出错误时
保留原版inactive/no-op，不猜步骤、不切成GRPO。验证只运行Student。
详见 [方法与适配边界](docs/METHOD.md)。

借用固定版本 [SDAR](https://github.com/ZJU-REAL/SDAR) 的原生WebShop环境、
prompt、history、parser、reward及Lucene搜索，不调用其训练算法。
每个实际游戏使用独立CPU子进程，避免跨rollout共享随机状态和商品会话。

## 3. 固定140题，与其他WebShop基线相同

[fixed_validation140.json](configs/fixed_validation140.json) 与
[task_split.json](configs/task_split.json) 和已发布GRPO、SDAR、RLSD仓库逐字节一致：

- beauty/electronics/fashion/garden/grocery各28题，140个不同目标商品。
- 按指令、属性和选项内容ID隔离，忽略生成的随机价格后缀；不靠局部序号隔离。
- 140题及其等价变体对应的145条原始记录全部排除，训练保留其余6765条。
- checkpoint、两个尺寸、续跑和独立评测共用同一140题、顺序、seed1000..1139。
- 验证分批4题控制CPU内存，但完整测140题；记录实际ID、清单及初始观察SHA。
- 报告成功数/140、成功率及独立的连续task score；原生终局score=1才算成功。

验证仍温度.4采样，固定题目不意味着输出/分数完全相同。
这是synthetic环境的自定义任务隔离验证，**不是官方human500或新商品zero-shot**。
清单SHA为706adb889929acc45bafd52fc144cd160b85ee7f752adf7b6dad8699f8c5ce4a。

## 4. 干净服务器安装

需要Linux x86-64、四张获配A100 **80GB**、git、uv和公开网络；
建议32逻辑CPU、64–128GiB主机RAM、100–150GiB空闲盘。这是部署预算，不是实测最低值。
原版uv.lock决定Torch/vLLM/CUDA版本，驱动必须匹配。
训练沿用原版Python>=3.13；WebShop另用Python3.10、CPU Torch2.8、Java11，
避免spaCy/Lucene依赖改坏训练栈。安装uv见其
[官方说明](https://docs.astral.sh/uv/getting-started/installation/)。

    git clone https://github.com/leiye302/hint-sd-webshop-qwen25-3b.git
    cd hint-sd-webshop-qwen25-3b

    # 先准备Java11；Ubuntu22.04示例，需要管理员权限：
    # sudo apt-get update && sudo apt-get install -y openjdk-11-jdk build-essential
    # export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

    bash scripts/setup.sh
    cp .env.example .env
    # 编辑自己获配的四张卡，不能占用其他作业。
    set -a; source .env; set +a

    .env-webshop/bin/python src/assets.py --download --nlp --index --verify
    .env-webshop/bin/python -m hint_webshop.env_smoke

    # 只下载本仓库尺寸的固定公开基模；不重新采集离线轨迹。
    .deps/hint-sd/.venv/bin/python -m hint_webshop.setup --download-model 3b
    python3 -m hint_webshop.setup --verify
    python3 scripts/check_release.py
    uv pip install --python .deps/hint-sd/.venv/bin/python pytest==8.3.5
    .deps/hint-sd/.venv/bin/python -m pytest tests -q
    bash scripts/train.sh --dry-run

商品JSON、指令、NLP wheel、源码和模型均固定revision/SHA。
原Google Drive匿名链接不可用时使用两个字节一致的公开HF镜像；
不伪称拿到作者历史SHA。使用原版索引程序，无外部购物网站、Chrome或API key。

CPU smoke读取已知目标进行搜索、选项、购买和奖励单测，
**不是Actor性能，不把答案发给策略**；同时准备6765条训练任务索引。
正式训练仍在线生成新轨迹，不依赖私有离线库。

## 5. 先做GPU smoke，再正式训练

    # 仅1更新、固定140中的前2题，另存smoke目录，不冒充正式成绩。
    bash scripts/train.sh --smoke

    # 确认成功后，在tmux/screen或调度器中正式运行。
    bash scripts/train.sh

首次GPU运行检查三路vLLM、LoRA同步、非空rollout、有效hint/选中步骤、
有限loss/grad、实际Student更新、EMA、完整存盘与退出释放。
inactive/no-op不能当成成功更新；CPU测试不能替代GPU验收。
兼容入口run_both.sh在本仓库只运行同一个3B，不会另起3B。

每个run有控制器排他锁，拒绝覆盖同名输出，只管理自建进程，
不整机清理或抢卡。日志/反馈/token证据只留服务器，不上传公共仓库。

    # 中断后先确认旧控制器/worker退出，再续跑。
    bash scripts/train.sh --resume

    bash scripts/eval.sh --eval-base --output "$PWD/runs/base_3b_fixed140"
    bash scripts/eval.sh --eval-checkpoint "$PWD/runs/hint_sd_3b/checkpoint-150" \
      --output "$PWD/runs/final_3b_fixed140"
    python3 -m hint_webshop.report

输出包含binding、日志/metrics、训练rollout/反馈、各次验证summary/token IDs、
checkpoint-140/150、final_adapter及completed.json。使用最终150步，不以验证集
最优模型替代最终结果。恢复要求模型/配方/源码/数据/文件SHA一致，并保留
Student/EMA、optimizer/scheduler/RNG和Trainer进度。新断点先同盘暂存，CRC/SHA验证、
原子发布后才清理旧断点；失败或损坏产物保留，不覆盖。vLLM随机状态和已执行游戏
会话不在原版断点中，不承诺与未中断训练逐位相同。

## 6. 检查和许可

GitHub Actions在干净公开Linux环境中检查原版KL/mask/EMA、原版Trainer CPU
backward/存盘恢复、四卡分配/三副本顺序、隐私来源，以及真实WebShop
prompt/search/options/buy/reward与独立CPU子进程。
**没有声称完成四卡CUDA/vLLM训练，没有预填模型成绩。**

包装代码及HINT-SD/SDAR为Apache-2.0，WebShop另有MIT许可；
1.5B模型Apache-2.0，3B为Qwen Research License。权重/商品资产不随Git分发。
见 [THIRD_PARTY.md](THIRD_PARTY.md)。请引用HINT-SD，并说明是WebShop/Qwen2.5适配，
不是作者报告的WebShop成绩。
