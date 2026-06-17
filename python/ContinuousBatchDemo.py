from dataclasses import dataclass
from typing import List, Deque
from collections import deque

# 模拟推理请求
@dataclass
class Request:
    req_id: int
    prompt: str
    target_gen_tokens: int  # 需要生成多少个输出token才结束
    generated_tokens: int = 0  # 当前已生成token数量

    def is_finished(self) -> bool:
        # 生成达到目标长度则结束
        return self.generated_tokens >= self.target_gen_tokens

# 模拟推理引擎（仅实现Continuous Batching调度）
class ContinuousBatchEngine:
    def __init__(self, max_running_seqs: int = 3):
        self.max_running_seqs = max_running_seqs  # 最大同时并发decode数量
        self.waiting_queue: Deque[Request] = deque()  # 等待入队的新请求
        self.running_batch: List[Request] = []  # 当前正在迭代的活跃请求

    def add_request(self, req: Request):
        """外部接口：提交新推理请求"""
        self.waiting_queue.append(req)
        print(f"[入队] 请求{req.req_id} 加入等待队列，需生成{req.target_gen_tokens}个token")

    def _prefill_new_req(self, req: Request):
        """模拟Prefill阶段：输入编码，初始化KV缓存（这里只打印日志）"""
        print(f"[Prefill] 请求{req.req_id} 编码输入prompt: {req.prompt}")
        return req

    def step(self):
        """核心：一轮token迭代（对应GPU一次decode step）"""
        print("\n===== 新一轮推理 Step 开始 =====")

        # 1. 模拟GPU并行：所有running请求各生成1个token
        for req in self.running_batch:
            req.generated_tokens += 1
            print(f"[Decode] 请求{req.req_id} 生成第{req.generated_tokens}个token")

        # 2. 剔除本轮已经完成的请求（连续批核心动作1）
        finished_reqs = [r for r in self.running_batch if r.is_finished()]
        self.running_batch = [r for r in self.running_batch if not r.is_finished()]
        for fin_req in finished_reqs:
            print(f"[完成] 请求{fin_req.req_id} 生成完毕，退出批次，释放KV资源")

        # 3. 从等待队列补充新请求，填满running最大并发（连续批核心动作2）
        free_slots = self.max_running_seqs - len(self.running_batch)
        if free_slots > 0 and len(self.waiting_queue) > 0:
            fill_num = min(free_slots, len(self.waiting_queue))
            for _ in range(fill_num):
                new_req = self.waiting_queue.popleft()
                self._prefill_new_req(new_req)
                self.running_batch.append(new_req)
                print(f"[新增] 请求{new_req.req_id} 加入运行批次，当前活跃数：{len(self.running_batch)}")

        print(f"Step结束 | 活跃请求数: {len(self.running_batch)} | 等待队列长度: {len(self.waiting_queue)}")

    def run_all(self):
        """循环执行step，直到所有请求全部处理完"""
        step_cnt = 0
        while len(self.running_batch) > 0 or len(self.waiting_queue) > 0:
            self.step()
            step_cnt += 1
        print(f"\n全部请求处理完成，总迭代步数：{step_cnt}")


if __name__ == "__main__":
    # 初始化引擎，最多同时跑3个decode请求
    engine = ContinuousBatchEngine(max_running_seqs=3)

    # 提交一批长短混合请求
    engine.add_request(Request(req_id=1, prompt="问题1", target_gen_tokens=3))  # 短请求，3步结束
    engine.add_request(Request(req_id=2, prompt="问题2", target_gen_tokens=10)) # 超长请求
    engine.add_request(Request(req_id=3, prompt="问题3", target_gen_tokens=2))  # 极短请求
    engine.add_request(Request(req_id=4, prompt="问题4", target_gen_tokens=4))  # 后面补充的新请求
    engine.add_request(Request(req_id=5, prompt="问题5", target_gen_tokens=5))

    # 持续迭代直到全部跑完
    engine.run_all()
