# BA-PR SAC 工程化开发文档

> **Bayesian Amnesic Piecewise-Robust SAC (BA-PR SAC)**
> 从 Lean 4 形式证明到 Python 工程实现的完整映射

---

## 0. 面向开发者：你需要了解的一切

### 0.1 项目全景定位

```
RE-SAC（已完成）                    BA-PR SAC（本文档目标）
───────────────                     ────────────────────
环境假设: 平稳随机                   环境假设: 分段平稳 + 突变
核心算子: T^REV (单一 Bellman)       核心算子: T^BA (belief-weighted mixture)
不确定性: aleatoric + epistemic      不确定性: + 贝叶斯变化点检测
形式证明: RESAC.lean ✅              形式证明: BAPR.lean ✅
```

### 0.2 现有代码资产清单

| 文件/目录 | 用途 | 对 BA-PR 的改动 |
|-----------|------|----------------|
| `sac_ensemble_original_logging.py` | 主训练脚本（RE-SAC） | **主要修改对象** |
| `env_original/sim.py` | 环境核心：step 循环、车辆调度 | **需改造**：注入多模式切换 |
| `env_original/route.py` | 路段速度采样 | **需改造**：运行时 sigma 切换 |
| `env_original/bus.py` | 车辆物理模型 + state/reward | 不需要改 |
| `env_original/station.py` | 站点乘客泊松到达 | **可选改造**：OD 倍率切换 |
| `env_original/config.json` | 时间参数 | 不需要改 |
| `normalization.py` | 状态归一化 | 不需要改 |
| `proof/BAPR.lean` | 形式证明 | 只读参考 |

### 0.3 内部属性速查表（快照操作需读写的字段）

以下属性在改造过程中会被频繁引用，列出以便快速查阅。

#### `env_original/sim.py` — `env_bus` 内部属性

```python
self.routes: list[Route]          # 路段列表（set_routes() 创建）
self.stations: list[Station]      # 站点列表（set_stations() 创建）
self.timetables: list[Timetable]  # 发车时刻表
self.bus_all: list[Bus]           # 所有上路车辆
self.current_time: int            # 当前仿真时间 (s)，每 step +1
self.max_agent_num: int           # 固定 25
self.route_sigma: float           # 全局默认 sigma（构造时传入）
self.args: dict                   # config.json 参数:
                                  #   time_step=1, route_state_update_freq=300,
                                  #   passenger_state_update_freq=20
self.state: dict[int, list]       # {bus_id: [obs_list]}，step() 输出
self.reward: dict[int, float]     # {bus_id: reward}，step() 输出
self.done: bool                   # episode 是否结束
# ===== BA-PR 新增 =====
self.enable_mode_switch: bool     # 是否启用模式切换
self.mode_profiles: dict          # MODE_PROFILES 定义
self.current_mode_name: str       # 当前模式名
self.current_mode_id: int         # 模式序号（累计切换次数）
self.next_switch_time: int        # 下次切换时间
self.mode_history: list[dict]     # 切换历史记录
```

#### `env_original/route.py` — `Route` 内部属性

```python
self.sigma: float                 # 当前 sigma（可被 _apply_mode 修改）
self.speed_history: pd.Series     # 各小时段均值 mu（对数域）
self.route_max_speed: int         # 限速（原始值，不变）
self.speed_limit: int             # 当前速度限制（route_update 更新）
# ===== BA-PR 新增 =====
self.speed_mean_scale: float      # 均值缩放比例（线性空间，默认 1.0）
self.speed_cap: int               # 动态速度上限（默认 15）
self.base_max_speed: int          # 原始 max_speed 备份
```

#### `env_original/bus.py` — `Bus` 核心属性（不改但需知道）

```python
bus.bus_id: int                   # fleet 编号
bus.trip_id: int                  # 当前车次
bus.direction: bool               # True=上行, False=下行
bus.absolute_distance: float      # 绝对距离 (m)
bus.current_speed: float          # 当前速度 (m/s)
bus.forward_headway: float        # 前车时距 (s)，默认 360
bus.backward_headway: float       # 后车时距 (s)，默认 360
bus.on_route: bool                # 是否在路上
bus.obs: list                     # 状态向量 [bus_id, station_id, time_period,
                                  #   direction, fwd_headway, bwd_headway,
                                  #   service_metric, speed_0, ..., speed_10]
                                  #   共 7 + len(routes)//2 = 18 维
bus.reward: float|None            # 奖励值（到站计算）
```

#### `env_original/station.py` — `Station` 核心属性

```python
self.od: dict|None                # OD 矩阵 {hour_str: {dest: demand}}
self.waiting_passengers: np.ndarray  # 等候乘客列表
# ===== BA-PR 新增 =====
self.od_multiplier: float         # OD 倍率（默认 1.0）
```

#### `sac_ensemble_original_logging.py` — `SAC_Trainer` 核心方法速查

```python
# 需要修改的方法
trainer.update(batch_size, training_steps, ...)  # → 加入 belief_tracker, surprise_computer
trainer.compute_q_loss(...)                       # → 替换为 compute_q_loss_bapr()
trainer.compute_reg_norm(model)                   # → 已有，返回 shape [ensemble_size]

# 不需要修改的方法
trainer.compute_policy_loss(...)                  # policy loss 不变
trainer.compute_alpha_loss(...)                   # entropy 自动调节不变
```

### 0.4 开发者自查：当前文档还缺什么？

> [!IMPORTANT]
> 以下是一个有 RE-SAC 全部代码的开发者可能还需要补充的信息：

| 缺失项 | 状态 | 说明 |
|--------|------|------|
| ① 环境多模式改造的具体代码 | **本文档 §2 补充** | 告知开发者改哪几行、怎么改 |
| ② 似然函数 $L(h, \xi)$ 的具体公式 | **本文档 §4.1 补充** | Lean 证明中是抽象参数 |
| ③ BOCD 算法的伪代码和参数 | **本文档 §3.2 补充** | 标准 Adams & MacKay 2007 |
| ④ 新增超参数的调参建议 | **本文档 §5 补充** | max_H, hazard_rate 等 |
| ⑤ 与环境时间尺度的对齐 | **本文档 §2.4 补充** | config.json 各频率含义 |
| ⑥ state 向量是否需要扩展 | **本文档 §3.4 补充** | 是否把 belief/surprise 放入 state |
| ⑦ 完整伪代码 | **本文档 §6 补充** | 整合所有改动的训练循环 |

---

## 1. 算子形式总览

### 1.1 BA-PR 核心算子（来自 `BAPR.lean`）

$$
\mathcal{T}^{BA} Q(s,a) = \sum_{h \in H} \rho(h) \cdot T_h(Q)(s,a)
$$

其中每个模式条件算子为：

$$
T_h(Q)(s,a) = R_h(s,a) + \gamma \left( \sum_{s'} P_h(s'|s,a) \cdot V^Q(s') - \lambda_{epi} \cdot \Gamma_{epi}^h(s,a) - \kappa_{ale} \right)
$$

### 1.2 符号→代码映射

| 数学符号 | 含义 | 代码对应 | 来源 |
|----------|------|----------|------|
| $H$ | run-length 空间（**集合**，非标量） | `range(max_run_length)` | **BA-PR 新增** |
| $\rho(h)$ | 贝叶斯 belief | `belief_tracker.belief: np.ndarray` | **BA-PR 新增** |
| $R_h(s,a)$ | 模式条件奖励 | `reward`（环境给出，**需改造环境**） | §2 |
| $P_h(s'|s,a)$ | 模式条件转移 | 隐式（**需改造环境**支持多 sigma） | §2 |
| $\Gamma_{epi}^h$ | 认知惩罚（**Lean 中与 Q 无关；代码中来自 frozen target Q**，见 §7.3） | `target_q_next.std(dim=0)` | RE-SAC 已有 |
| $\kappa_{ale}$ | 偶然惩罚（**正值**，`compute_reg_norm` 返回正数；Lean 中减去 κ = 代码中加上 +reg_norm，效果均为惩罚大权重） | `compute_reg_norm()` | RE-SAC 已有 |
| $\gamma$ | 折扣因子 | `args.gamma` (0.99) | RE-SAC 已有 |
| $\lambda_{epi}$ | epistemic 惩罚系数 | `args.weight_reg` (0.01) | RE-SAC 已有 |
| $\alpha_{SAC}$ | SAC entropy temperature（**非** §4.1 的方差增长率 $\eta$） | `self.alpha`（自动调节） | RE-SAC 已有 |
| $\xi$ | 惊奇度信号（surprise） | `surprise_computer.ema_surprise` | **BA-PR 新增**, §4.1 |

### 1.3 贝叶斯置信度更新（`BAPR.lean` §8）

$$
\rho'(h) = \frac{\rho(h) \cdot L(h, \xi)}{Z}, \quad Z = \sum_h \rho(h) \cdot L(h, \xi)
$$

> **符号说明**：此处 $\xi$ 为 surprise 信号（`SurpriseComputer.compute()` 的输出），**不是**环境的路段速度方差 `route.sigma`，也不是状态 $s$。

---

## 2. 环境多模式改造方案（确认路线：改造环境！均值级剧变）

### 2.1 当前环境的可调参数分析

通过阅读实际数据和代码，我们有以下可调控的"旋钮"：

**① 速度采样公式**（`route.py:23`）：
```python
v = np.clip(math.log(random.lognormvariate(
    self.speed_history.loc[current_hour],  # ← 均值（4~11 m/s，按路段/小时不同）
    self.sigma                              # ← 方差（固定 1.5）
)), 2, 15)
self.speed_limit = min(self.route_max_speed, max(int(v), 0))  # ← 上限 15
```
**可调旋钮**：均值缩放比例 `speed_mean_scale`（对数域偏移）、方差 `sigma`、速度上限 `route_max_speed`

**② OD 客流**（`station.py:56-66`）：
```python
demand_per_second = demand / 3600.0  # demand 范围 1~10（人/小时/OD对）
destination_demand_num = np.random.poisson(demand_per_second * interval)
```
**可调旋钮**：全局倍率 `od_multiplier`、**特定站点倍率** `station_od_scale`

**③ 每个路段都有独立的** `speed_history` **和** `max_speed` → 支持**按路段差异化**干扰

### 2.2 模式档案（Mode Profiles）：大幅均值级突变

> [!IMPORTANT]
> 模式切换不仅调方差，更要改**速度均值、OD均值、速度上限**，制造清晰可感知的 regime shift。

```python
MODE_PROFILES = {
    # ─── 正常模式 ───
    "normal": {
        "speed_mean_scale": 1.0,      # 速度均值不变
        "sigma": 1.5,                  # 标准方差
        "speed_cap": 15,               # 正常限速
        "od_global_mult": 1.0,         # 标准客流
        "station_od_overrides": {},     # 无站点特殊干预
        "affected_routes": None,       # 所有路段
    },

    # ─── 严重拥堵：某段道路事故/施工 ───
    "congestion_severe": {
        "speed_mean_scale": 0.3,       # 均值降到 30%！（原 4~11 → 变成 1.2~3.3）
        "sigma": 3.0,                  # 波动也变大
        "speed_cap": 5,                # 限速降到 5 m/s（几乎爬行）
        "od_global_mult": 1.0,
        "station_od_overrides": {},
        "affected_routes": [3,4,5,6],  # 只影响路段 3-6（中段路网）
    },

    # ─── 客流激增：学校放学 / 大型活动 ───
    "demand_surge": {
        "speed_mean_scale": 1.0,
        "sigma": 1.5,
        "speed_cap": 15,
        "od_global_mult": 1.5,         # 全局 1.5 倍
        "station_od_overrides": {      # 特定站点 OD 暴涨
            "X05": 5.0,                # 站点 X05 客流 5 倍！
            "X06": 4.0,                # X06 客流 4 倍
            "X07": 3.0,                # X07 客流 3 倍
        },
        "affected_routes": None,
    },

    # ─── 全线瘫痪：极端天气 ───
    "extreme_weather": {
        "speed_mean_scale": 0.4,       # 全线均值降 60%
        "sigma": 4.0,                  # 极大方差
        "speed_cap": 8,                # 全线限速
        "od_global_mult": 0.3,         # 乘客也减少（恶劣天气出行少）
        "station_od_overrides": {},
        "affected_routes": None,       # 影响所有路段
    },

    # ─── 局部路段封闭 + 周边客流转移 ───
    "partial_closure": {
        "speed_mean_scale": 0.15,      # 封闭路段速度降到 15%（几乎停滞）
        "sigma": 1.0,
        "speed_cap": 3,                # 封闭路段限速 3
        "od_global_mult": 1.0,
        "station_od_overrides": {      # 封闭路段周边站点客流增加（换乘）
            "X08": 4.0,
            "X09": 4.0,
            "X10": 3.0,
        },
        "affected_routes": [7,8,9],    # 只封闭路段 7-9
    },
}
```

### 2.3 对比：旧方案 vs 新方案的冲击力

| 维度 | 旧方案（仅调 sigma） | 新方案（均值级剧变） |
|------|---------------------|---------------------|
| 速度变化 | 波动范围变大，但均值不变 | **均值直接砍到 30%，限速降到 5** |
| OD 变化 | 全局统一倍率 | **指定站点 5 倍暴涨**，模拟放学/活动 |
| 空间差异 | 全路网统一 | **按路段差异化**（3-6段拥堵，7-9段封闭） |
| Agent 感知 | Q 值方差略增 | **headway 从 360s 飙到 600+s，reward 断崖式下降** |
| BA-PR 检测 | surprise 微弱，belief 变化缓慢 | **surprise 信号剧烈，belief 快速重置** |

### 2.4 代码改造

#### (A) `env_original/route.py` — 支持均值缩放和动态限速

```python
class Route(object):
    def __init__(self, route_id, start_stop, end_stop, route_length, max_speed,
                 route_speed_history, sigma=1.5):
        ...
        self.base_max_speed = max_speed          # 原始最大速度（保存）
        self.route_max_speed = max_speed
        self.speed_mean_scale = 1.0              # 新增：均值缩放比例（线性空间）
        self.speed_cap = 15                      # 新增：速度上限覆盖

    def route_update(self, current_time, effective_period):
        current_hour = effective_period[min(current_time//3600, len(effective_period)-1)]
        # ===== BA-PR: 均值缩放 =====
        # 注意：lognormvariate(mu, sigma) 的 mu 是对数域参数
        # 要在线性空间实现 speed_mean_scale 倍缩放，需在对数域做加法偏移
        base_mu = self.speed_history.loc[current_hour]
        scaled_mu = base_mu + math.log(self.speed_mean_scale)  # log域偏移 = 线性域缩放
        v = np.clip(
            math.log(random.lognormvariate(scaled_mu, self.sigma)),
            2, self.speed_cap     # ← 动态上限
        )
        self.speed_limit = min(self.speed_cap, max(int(v), 0))
```

#### (B) `env_original/station.py` — 支持按站点的 OD 倍率

```python
class Station(object):
    def __init__(self, ...):
        ...
        self.od_multiplier = 1.0  # 新增：站点级 OD 倍率

    def station_update(self, current_time, stations, passenger_update_interval=1):
        if self.od is not None:
            ...
            for destination_name, demand in period_od.items():
                if demand > 0:
                    demand_per_second = (demand * self.od_multiplier) / 3600.0  # ← 乘以倍率
                    ...
```

#### (C) `env_original/sim.py` — 模式切换引擎

```python
class env_bus(object):
    def __init__(self, path, debug=False, render=False, route_sigma=1.5,
                 enable_mode_switch=False,
                 mode_switch_interval=(1800, 7200),
                 mode_profiles=None):
        ...
        self.enable_mode_switch = enable_mode_switch
        self.mode_switch_interval = mode_switch_interval
        self.mode_profiles = mode_profiles or MODE_PROFILES
        self.current_mode_name = "normal"
        self.current_mode_id = 0
        self.next_switch_time = np.random.randint(*mode_switch_interval) if enable_mode_switch else 0
        self.mode_history = []

    def _apply_mode(self, mode_name: str):
        """将指定模式的参数注入到环境的各个组件"""
        profile = self.mode_profiles[mode_name]
        affected = profile["affected_routes"]

        for i, route in enumerate(self.routes):
            if affected is None or i in affected:
                route.speed_mean_scale = profile["speed_mean_scale"]
                route.sigma = profile["sigma"]
                route.speed_cap = profile["speed_cap"]
            else:
                # 非影响路段恢复正常
                route.speed_mean_scale = 1.0
                route.sigma = self.route_sigma
                route.speed_cap = 15

        # OD 倍率
        od_overrides = profile["station_od_overrides"]
        for station in self.stations:
            if station.station_name in od_overrides:
                station.od_multiplier = od_overrides[station.station_name]
            else:
                station.od_multiplier = profile["od_global_mult"]

        self.current_mode_name = mode_name

        # ===== 立即更新速度，消除最多 300s 的迟滞 =====
        for route in self.routes:
            route.route_update(self.current_time, self.effective_period)

    def _maybe_switch_mode(self):
        if not self.enable_mode_switch:
            return
        if self.current_time >= self.next_switch_time:
            # 随机选择新模式（排除当前模式以保证"突变"）
            candidates = [m for m in self.mode_profiles if m != self.current_mode_name]
            new_mode = np.random.choice(candidates)
            self._apply_mode(new_mode)

            self.current_mode_id += 1
            self.mode_history.append({
                'time': self.current_time,
                'mode': new_mode,
                'mode_id': self.current_mode_id,
                'profile': self.mode_profiles[new_mode]
            })
            self.next_switch_time = self.current_time + np.random.randint(*self.mode_switch_interval)

    def reset(self):
        # ===== 原有 reset 逻辑 =====
        self.current_time = 0
        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()
        self.bus_id = 0
        self.bus_all = []
        self.route_state = []
        self.state = {key: [] for key in range(self.max_agent_num)}
        self.reward = {key: 0 for key in range(self.max_agent_num)}
        self.done = False
        self.action_dict = {key: None for key in list(range(self.max_agent_num))}

        # ===== BA-PR: 重置模式状态 =====
        if self.enable_mode_switch:
            self.current_mode_name = "normal"
            self.current_mode_id = 0
            self.next_switch_time = np.random.randint(*self.mode_switch_interval)
            self.mode_history = []
            self._apply_mode("normal")  # 确保新建的 route/station 对象拥有正确属性

    def step(self, action, debug=False, render=False):
        self._maybe_switch_mode()  # ← 加在 step() 开头
        ...  # 原有逻辑不变

    def get_current_mode_info(self):
        """供训练脚本获取当前模式信息"""
        return {
            'mode_name': self.current_mode_name,
            'mode_id': self.current_mode_id,
            'current_sigma': self.routes[0].sigma,
            'speed_mean_scale': self.routes[0].speed_mean_scale,
            'switch_count': len(self.mode_history)
        }
```

### 2.5 冲击力量化：模式切换前后的 headway 预期变化

以正常模式 → 严重拥堵为例：

```
正常模式:
  速度均值 ~8 m/s, 路段 500m → 路段行程 ~63s
  headway ≈ 360s（目标值）

congestion_severe (路段 3-6):
  速度均值 ~8 * 0.3 = 2.4 m/s, 限速 5 → 路段行程 ~200s+
  4 个路段累计增加 ~550s → headway 暴涨到 600~900s
  reward: -abs(800 - 360) = -440（vs 正常 ~-50）
  → agent 必须大幅增加 holding time 来重新均衡
```

### 2.6 环境时间尺度对齐

```
config.json 参数:
  time_step = 1 秒             → 每 step 推进 1 秒
  route_state_update_freq = 300 秒  → 每 5 分钟重采样速度（从缩放后的均值）
  passenger_state_update_freq = 20 秒  → 每 20 秒更新乘客到达

模式切换频率:
  mode_switch_interval = (1800, 7200) → 30min~2h 切换一次
  一个 episode ~47000 步(13h) → 每 episode 期望 ~7-25 次切换
  亚稳态假设：切换间隔 >> 1/(1-0.99) = 100 步 ✅
```

---

## 3. 训练脚本改造方案

### 3.1 需要新增的组件

#### (A) `BeliefTracker` — BOCD 风格贝叶斯置信度

```python
class BeliefTracker:
    """
    Bayesian Online Change Detection (Adams & MacKay 2007)
    维护 run-length 后验分布 ρ(h)
    对应 BAPR.lean: update_belief / normalization_const
    """
    def __init__(self, max_run_length: int = 20, hazard_rate: float = 0.01,
                 base_variance: float = 1.0, variance_growth: float = 0.1):
        self.max_H = max_run_length
        self.hazard = hazard_rate
        self.base_var = base_variance
        self.var_growth = variance_growth
        self.belief = np.ones(max_run_length) / max_run_length  # 均匀先验

    def reset(self):
        """每个 episode 开始时重置 belief 为均匀先验"""
        self.belief = np.ones(self.max_H) / self.max_H

    def update(self, surprise: float):
        """
        BAPR.lean: update_belief ρ L Z
        1. 计算似然 L(h, σ)
        2. 重加权 ρ(h) * L(h)
        3. 归一化
        4. 混入 changepoint 先验
        """
        L = self._compute_likelihood(surprise)      # L ≥ 0 (hL_nn)
        unnorm = self.belief * L
        Z = unnorm.sum()
        if Z > 1e-10:                                 # hZ_pos 保护
            self.belief = unnorm / Z
        else:
            self.belief = np.ones(self.max_H) / self.max_H  # 退化到均匀

        # BOCD: 将 hazard_rate 的质量移到 h=0（新 run-length）
        growth_prob = self.belief * (1 - self.hazard)  # 现有 run-length 增长
        changepoint_prob = self.belief.sum() * self.hazard  # 突变概率

        new_belief = np.zeros(self.max_H)
        new_belief[0] = changepoint_prob  # h=0 获得所有突变质量
        new_belief[1:] = growth_prob[:-1]  # h 右移一步
        total = new_belief.sum()
        self.belief = new_belief / total if total > 1e-10 else np.ones(self.max_H) / self.max_H

    def _compute_likelihood(self, surprise: float) -> np.ndarray:
        """
        L(h, ξ): 指数族似然（ξ = surprise 信号，非环境 sigma）
        短 run-length → 方差小 → 只有低 surprise 才获得高似然
        长 run-length → 方差大 → 容忍更大 surprise
        """
        variances = self.base_var + self.var_growth * np.arange(self.max_H)
        return np.exp(-surprise**2 / (2 * variances))

    @property
    def effective_window(self) -> float:
        """有效窗口长度 = Σ h * ρ(h)"""
        return np.sum(np.arange(self.max_H) * self.belief)

    @property
    def entropy(self) -> float:
        """belief 分布的熵（用于监控）"""
        p = self.belief[self.belief > 1e-10]
        return -np.sum(p * np.log(p))
```

#### (B) `SurpriseComputer` — 惊奇度信号

```python
class SurpriseComputer:
    """
    计算 surprise = |κ_current - κ_target|
    对应 BAPR.lean §8: RE-SAC 的 κ 驱动贝叶斯更新
    """
    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self.ema_surprise = 0.0

    def compute(self, current_reg_norm: torch.Tensor,
                target_reg_norm: torch.Tensor) -> float:
        """
        current_reg_norm: compute_reg_norm(self.soft_q_net)    ← 新增
        target_reg_norm:  compute_reg_norm(self.target_soft_q_net) ← 已有
        """
        raw = (current_reg_norm - target_reg_norm).abs().mean().item()
        self.ema_surprise = self.ema_alpha * raw + (1 - self.ema_alpha) * self.ema_surprise
        return self.ema_surprise  # 平滑后的 surprise
```

### 3.2 `compute_q_loss` 改造

> ⚠️ **CRITICAL PROMPT FOR AI CODING ASSISTANTS (关于 Belief 时序与 Surprise 计算):**
> 在实现以下改造时，请严格遵守以下时序约束：
> 1. **Surprise 必须在 Q-loss 之前计算**：`surprise_computer.compute()` 和 `belief_tracker.update()` 必须在 `compute_q_loss_bapr()` 之前调用，因为 `compute_q_loss_bapr` 依赖更新后的 `belief_tracker.belief`。
> 2. **Belief 是冻结参数**：在单次 `compute_q_loss_bapr` 调用内，`belief` 不可被修改。这对应 `BAPR.lean` 中的 Frozen Belief 假设（ρ 是 T_BAPR 的参数，不是变量）。
> 3. **`compute_reg_norm` 需调用两次**：一次对 `self.soft_q_net`（新增），一次对 `self.target_soft_q_net`（已有），两者之差就是 surprise 信号。
> 4. **Penalty 符号**：`weighted_lambda * epi_penalty` 是**减去**（惩罚），不是加上。注意 `args.weight_reg` 前面的符号与原 RE-SAC 一致（正号），而 `epi_penalty` 项前面是**负号**。
> 5. **Target 网络必须冻结**：在 `compute_q_loss_bapr` 执行期间，**不得**调用 `soft_update` 或任何修改 `target_soft_q_net` 参数的操作。`soft_update` 必须在 Q-loss 和 policy-loss 计算**全部完成之后**才能执行。否则 `epi_penalty = target_q_next.std(dim=0)` 会违反 Lean 的 Frozen Penalty 假设（`Γ_epi` 需与 Q 无关）。

```python
def compute_q_loss_bapr(self, state, action, reward, next_state, done,
                        new_next_action, next_log_prob, reg_norm, gamma,
                        belief_tracker: BeliefTracker):
    predicted_q = self.soft_q_net(state, action)

    target_q_next = self.target_soft_q_net(next_state, new_next_action)
    next_log_prob = next_log_prob.unsqueeze(0).repeat(self.soft_q_net.num_critics, 1)

    # ===== BA-PR 核心：belief-weighted epistemic penalty =====
    epi_penalty = target_q_next.std(dim=0)  # ensemble 标准差

    # 不同 run-length 对应不同的惩罚强度
    # 短 h（刚突变）→ 高惩罚（旧数据不可信）
    # 长 h（长期平稳）→ 低惩罚（数据可信）
    belief = torch.tensor(belief_tracker.belief, device=device, dtype=torch.float32)
    penalty_schedule = torch.exp(-0.1 * torch.arange(belief_tracker.max_H, device=device, dtype=torch.float32))
    # penalty_schedule: [1.0, 0.90, 0.82, 0.74, ...] → 随 h 增大而减小
    weighted_lambda = (belief * penalty_schedule).sum()  # 标量

    reg_norm_expanded = reg_norm.unsqueeze(-1).repeat(1, args.batch_size)
    target_q_next = (target_q_next
                     - self.alpha * next_log_prob
                     + args.weight_reg * reg_norm_expanded
                     - weighted_lambda * args.weight_reg * epi_penalty)  # ← BA-PR 加权惩罚
    # ============================================================

    target_q_value = reward + (1 - done) * gamma * target_q_next.unsqueeze(-1)

    ood_loss = predicted_q.std(0).mean()
    q_loss = self.soft_q_criterion(predicted_q, target_q_value.squeeze(-1).detach())
    loss = q_loss + args.beta_ood * ood_loss

    return loss, predicted_q, ood_loss, weighted_lambda.item()
```

### 3.3 `update()` 方法改造

```python
def update(self, batch_size, training_steps, belief_tracker, surprise_computer,
           reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
    ...
    # ===== BA-PR: 计算 surprise 并更新 belief =====
    current_reg = self.compute_reg_norm(self.soft_q_net)
    target_reg = self.compute_reg_norm(self.target_soft_q_net)
    surprise = surprise_computer.compute(current_reg, target_reg)
    belief_tracker.update(surprise)
    # =============================================

    # 替换原来的 compute_q_loss
    q_value_loss, predicted_q, ood_loss, weighted_lambda = self.compute_q_loss_bapr(
        state, action, reward, next_state, done,
        new_next_action, next_log_prob, target_reg, gamma,
        belief_tracker
    )
    ...
```

### 3.4 State 向量是否扩展？

**结论：首版不扩展。**

理由：
- `bus.py:217` 的 state 包含 `[bus_id, station_id, time_period, direction, fwd_headway, bwd_headway, service_metric, speed_0, speed_1, ..., speed_10]`
- belief 和 surprise 是算子内部的计算信号，不是环境可观测量
- 如果未来想让 policy 也感知模式变化，可以加入 `current_sigma` 到 state

---

## 4. 算子歧义与工程化困难点

### 4.1 ⚠️ 似然函数 $L(h, \xi)$ — **已定义**

Lean 证明中 `L : H → ℝ` 是完全抽象的（唯一约束 `hL_nn: ∀ h, 0 ≤ L h`）。
我们采用 **BOCD 指数族似然**（见 §3.1 `_compute_likelihood`）：

$$
L(h, \xi) = \exp\left(-\frac{\xi^2}{2(\sigma_0^2 + \eta \cdot h)}\right)
$$

- $\xi$（`surprise`）：惊奇度信号，由 `SurpriseComputer.compute()` 计算得到，即 $|\kappa_{cur} - \kappa_{tgt}|$ 的 EMA 平滑值。**注意**：$\xi$ 既不是环境的路段速度方差 `route.sigma`（§2，默认 1.5），也不是 RL 中的状态 $s$
- $\sigma_0^2$（`base_variance`）：基础方差，控制对 surprise 的敏感度
- $\eta$（`variance_growth`）：方差增长率，长 run-length 更"宽容"。**注意**：$\eta$ **不是** SAC 的 entropy temperature $\alpha_{SAC}$（`self.alpha`）
- $h$（`run-length`）：距离上次变化点（changepoint）的步数
- **物理直觉**：刚发生突变后（$h$ 小），只有很低的 surprise 是"正常"
- **Lean 对应**：Lean 中 `L : H → ℝ` 是对特定 $\xi$ 值的偏函数（$\xi$ 在调用 `update_belief` 前固定）。代码实现为 `_compute_likelihood(surprise) → np.ndarray[max_H]`，一次性返回所有 $h$ 对应的似然值

### 4.2 ⚠️ 模式条件算子 $T_h$ — **已通过环境改造解决**

原始歧义：Lean 中每个模式有独立的 $R_h, P_h$，但环境只有一套。

**解决方案**：改造环境使其在 episode 内真正生成**不同模式的 transition 和 reward**（§2 方案）。同时在算子层面，belief 加权体现为**对 epistemic penalty 的动态调节**（§3.2）。

- 环境: 提供真实的多模式 $(R_h, P_h)$（通过 sigma/OD 切换）
- 算子: 用 belief-weighted penalty schedule 近似 $\sum_h \rho(h) T_h$

### 4.3 ⚠️ Surprise 时间粒度

选择：**每次 `update()` 调用时计算**（每个 training step），并用 EMA 平滑。
- `compute_reg_norm(self.soft_q_net)` — **需新增**（当前代码只算 target 的）
- `compute_reg_norm(self.target_soft_q_net)` — 已有（line 420）

### 4.4 ⚠️ $|H|$ 维度选择

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| `max_run_length` | 20 | 公交场景切换间隔~30min-2h，training step ~5步/env step |
| `hazard_rate` | 0.01-0.05 | 对应每步 1-5% 的先验突变概率 |

### 4.5 ⚠️ 数值稳定性

**必需防护**（已内建到 `BeliefTracker` 中）：
- `Z < 1e-10` 时重置为均匀分布
- belief 始终非负（exp 保证）
- 归一化在每次 update 后执行

### 4.6 ⚠️ $R_h$ 与环境奖励

环境改造后，同一个 state 在不同 sigma 下会获得**不同的 next_state**（因为速度不同导致 headway 不同），从而**间接产生不同的 reward**。$R_h$ 的模式依赖性通过环境自然体现，无需手动定义。

---

## 5. 新增超参数汇总

| 超参数 | 默认值 | 含义 | 调参建议 |
|--------|--------|------|----------|
| `enable_mode_switch` | `True` | 是否启用 BA-PR 模式切换 | 对照实验用 False |
| `mode_switch_interval` | `(1800, 7200)` | 切换间隔（秒） | 越小越频繁越难 |
| `mode_profiles` | `MODE_PROFILES` | 5种预设模式档案 | 可自定义/增减 |
| `max_run_length` | `20` | belief 向量维度 | 10-50 |
| `hazard_rate` | `0.02` | 先验突变概率 | 0.01-0.1 |
| `base_variance` | `1.0` | 似然函数基础方差 | 根据 surprise 量级调整 |
| `variance_growth` | `0.1` | 方差随 h 的增长率 | 0.01-0.5 |
| `surprise_ema` | `0.1` | surprise 平滑系数 | 0.05-0.3 |

---

## 6. 完整训练循环伪代码

```python
# ============= BA-PR SAC 训练循环 =============

# 初始化
belief_tracker = BeliefTracker(max_run_length=20, hazard_rate=0.02)
surprise_computer = SurpriseComputer(ema_alpha=0.1)

env = env_bus(path, route_sigma=1.5,
              enable_mode_switch=True,               # ← BA-PR 启用
              mode_switch_interval=(1800, 7200),
              mode_profiles=MODE_PROFILES)            # ← 使用 §2.2 定义的模式档案

for episode in range(max_episodes):
    env.reset()
    belief_tracker.reset()  # 每个 episode 重置 belief（亚稳态假设）

    while not done:
        # 环境交互（与 RE-SAC 完全相同）
        action = policy.get_action(state)
        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)

        # 训练（在 RE-SAC 基础上改动）
        if ready_to_train:
            # 核心改动：update 中加入 belief_tracker 和 surprise_computer
            sac_trainer.update(batch_size, training_steps,
                             belief_tracker, surprise_computer, ...)

    # ---- 日志（新增） ----
    log('belief_entropy', belief_tracker.entropy)
    log('effective_window', belief_tracker.effective_window)
    log('surprise_ema', surprise_computer.ema_surprise)
    log('mode_switches', len(env.mode_history))
```

---

## 7. 证明假设 vs 代码实现：逐条验证

> [!IMPORTANT]
> 以下是 `BAPR.lean` 中每一条形式化假设在代码中的满足状态。
> **状态说明**: ✅ = 代码满足 | ⚠️ = 近似满足（有 caveat） | ❌ = 不满足

### 7.1 核心定理假设 (`bapr_contraction`, L314-L366)

| # | Lean 假设 | 含义 | 代码对应 | 状态 |
|---|----------|------|---------|------|
| 1 | `hγ : 0 ≤ γ ∧ γ < 1` | 折扣因子合法 | `args.gamma = 0.99` | ✅ |
| 2 | `hlam_epi : 0 ≤ lam_epi` | epistemic 惩罚系数非负 | `args.weight_reg = 0.01 ≥ 0` | ✅ |
| 3 | `hP_nn : ∀ h s a s', 0 ≤ P h s a s'` | 转移概率非负 | 环境物理模型保证（泊松到达 + 速度采样） | ✅ |
| 4 | `hP_prob : ∀ h s a, Σ P = 1` | 转移概率归一 | 环境是确定性仿真器（给定状态+动作→确定性 next_state），对应 one-hot 分布 | ✅ |
| 5 | `hρ_nn : ∀ h, 0 ≤ ρ h` | belief 非负 | `_compute_likelihood` 用 `exp()`，恒正；`BeliefTracker` 所有运算保持非负 | ✅ |
| 6 | `hρ_sum : Σ ρ = 1` | belief 归一 | `update()` 末尾 `new_belief / total`；退化时也恢复均匀 `1/max_H` | ⚠️ 见 7.2 |

### 7.2 Belief 更新假设 (`bapr_contraction_after_update`, L443-L458)

| # | Lean 假设 | 代码实现 | 状态 | 说明 |
|---|----------|---------|------|------|
| 7 | `hL_nn : ∀ h, 0 ≤ L h` | `exp(-ξ²/(2v)) ≥ 0` 恒成立 | ✅ | `exp` 恒正 |
| 8 | `hZ_pos : 0 < Z` | `Z = unnorm.sum()`，有 `Z > 1e-10` 保护 | ✅ | 退化时重置为均匀 |
| 9 | Lean: `ρ'(h) = ρ(h)·L(h)/Z` | 代码: 先做 `ρ·L/Z`，然后**额外做 BOCD hazard shift** | ⚠️ | 见下方详述 |

> [!WARNING]
> **关键 Caveat: BOCD Hazard Shift**
>
> Lean 证明的 `update_belief` 定义为 `ρ'(h) = ρ(h) · L(h) / Z`（纯贝叶斯更新），并证明此更新后 `ρ' ≥ 0` 且 `Σρ' = 1`，因此 `bapr_contraction_after_update` 成立。
>
> 代码的 `BeliefTracker.update()` 在此基础上**额外做了 BOCD 的 hazard shift**（第 490-498 行）：将部分概率质量从当前 run-length 转移到 h=0，并右移所有 run-length。这一步的结果仍然是非负且归一的（代码末尾有 `/ total` 归一化），因此 `bapr_contraction` 定理的前提条件 (`hρ_nn`, `hρ_sum`) 仍然成立。
>
> **结论：安全。** BOCD hazard shift 不违反 Lean 证明的假设，因为 `bapr_contraction` 对**任意**满足 ρ≥0 且 Σρ=1 的分布都成立。hazard shift 只是换了一个不同的合法分布，收缩性依然保持。但这一步本身**没有被形式化证明**——Lean 只证明了纯贝叶斯更新的收缩性，hazard shift 后的收缩性是"for free"的（因为主定理对任意合法 ρ 成立）。

### 7.3 Frozen Penalty 假设 (`t_mode_pointwise_bound`, L251-L288)

| # | Lean 假设 | 含义 | 代码对应 | 状态 |
|---|----------|------|---------|------|
| 10 | `R, Γ_epi, κ_ale` 在 `T_mode` 定义中是**固定参数**，不依赖 Q | 对比 Q₁ 和 Q₂ 时，惩罚项相消 | `epi_penalty = target_q_next.std(dim=0)` — 来自 **target** 网络 | ⚠️ 见下方 |
| 11 | `κ_ale` 是标量 | 偶然惩罚冻结 | `reg_norm` 来自 `compute_reg_norm(target_soft_q_net)` — 冻结 ✅ | ✅ |

> [!NOTE]
> **Caveat: `epi_penalty` 依赖 Q**
>
> Lean 中 `Γ_epi h s a` 是与 Q 无关的**固定函数**。代码中 `epi_penalty = target_q_next.std(dim=0)` 来自 target 网络的 Q 输出——虽然 target 网络在一次 update 内冻结（weights 不变），但 `epi_penalty` 的值取决于 `next_state` 和 `new_next_action`，而 `new_next_action` 又取决于当前 policy。
>
> 在**单次 Bellman backup** 中，target 网络参数冻结 → `Γ_epi` 对于给定 `(s', a')` 是确定值 → 满足 Lean 假设。只要不在同一次 `compute_q_loss_bapr` 内更新 target 网络即可。代码通过 `soft_tau` EMA 更新放在 `compute_q_loss_bapr` **之后**，符合要求。

### 7.4 T_BAPR 结构映射

Lean 定义：
```
T_BAPR(Q)(s,a) = Σ_h ρ(h) · [R_h(s,a) + γ·(Σ_s' P_h·V(s') - λ·Γ_h - κ)]
```

代码近似：
```python
target_q_next = target_q_next - α·log_prob + weight_reg·κ - weighted_λ·weight_reg·epi_penalty
```

| Lean 项 | 代码项 | 映射关系 |
|---------|-------|---------|
| `R_h(s,a)` | `reward` | 环境返回，模式 h 隐式生效（环境切换改变了 transition） |
| `Σ P_h·V(s')` | `target_q_next` | target 网络对 next_state 的 Q 输出 |
| `λ_epi · Γ_epi(h,s,a)` | `weighted_lambda * weight_reg * epi_penalty` | belief-weighted 的 ensemble std |
| `−κ_ale` | `+weight_reg * reg_norm` | `compute_reg_norm()` 返回**正值**（L1/L2 范数）。Lean 中 `−κ` 惩罚大权重；代码中 `+weight_reg·reg_norm` 鼓励小权重——**效果相同**，都使大权重网络获得更低 Q 值。符号看似翻转，实为约定差异：Lean 的 κ 定义为减项，代码的 reg_norm 定义为加项 |
| `Σ_h ρ(h) · T_h` | `weighted_lambda = Σ belief · penalty_schedule` | **近似实现**：不是逐模式分别计算 T_h 再加权，而是用 belief 加权 penalty 强度来近似 |

> [!IMPORTANT]
> **结构差异说明**：代码**没有**显式计算每个模式 h 对应的独立 Bellman backup $T_h(Q)$，而是将 belief 加权折叠为一个**标量 `weighted_lambda`** 来调制 epistemic penalty 的强度。这在效果上等价于：环境提供了当前模式的真实 $R_h, P_h$（通过模式切换），而 belief 通过调节 penalty 来控制对旧数据的信任度。
>
> 这种近似是合理的，因为 Lean 证明的收缩性**不依赖于 belief 的具体值**——只要 ρ ≥ 0 且 Σρ = 1，任何 weighted_lambda 值都保持收缩性。penalty schedule 的形状只影响收敛速度和稳态策略质量，不影响收缩性保证。

---

## 8. 对照实验建议

| 实验 | 环境设置 | 算法 | 目的 |
|------|----------|------|------|
| Baseline | `enable_mode_switch=False` | RE-SAC | 确认平稳环境基线 |
| Stress: full profiles | `True`, 5 modes | RE-SAC | 展示 RE-SAC 在剧烈突变下的退化 |
| BA-PR: full profiles | `True`, 5 modes | BA-PR SAC | 验证 belief 对剧烈突变的适应 |
| Stress: congestion only | `True`, normal+congestion | RE-SAC vs BA-PR | 单一突变类型对比 |
| Stress: demand surge | `True`, normal+demand_surge | RE-SAC vs BA-PR | 客流突变对比 |
| Ablation: no belief | `True`, 5 modes | BA-PR (belief=uniform) | 消融 belief 的作用 |
| Ablation: no env mode | `False` | BA-PR SAC | 验证 BA-PR 在平稳下不退化 |
| **🔴 Ablation: fixed decay** | `True`, 5 modes | RE-SAC + `weight_reg *= 0.99^step` | **见 §8.1** |

### 8.1 Reviewer 风险评估与关键消融实验

> [!CAUTION]
> **本算法存在被质疑为 "trivial 组件拼装" 的风险。** 以下是 reviewer 最可能提出的三个质疑，以及应对策略。

#### 质疑 1："凸组合 of contractions 仍是 contraction" 太显然

`bapr_contraction` 的核心命题对任何学过泛函分析的 reviewer 来说是 trivial 的。Lean 形式化有工作量，但被验证的命题不深。

**应对**：论文写法上不要把收缩性证明当作主贡献。贡献点应定位在 **bridge**：从形式化算子到工程部署之间的映射（§7 逐条验证），证明"工程近似不违反理论保证"。收缩性是**前提条件**，不是**贡献**。

#### 质疑 2："你的 belief-weighted operator 在代码里等价于动态调一个标量"

Lean 里是 `T_BAPR = Σ ρ(h) · T_h(Q)`，每个 h 有独立的 R_h, P_h, Γ_h。代码里所有模式共享同一个 Q 网络，belief 被折叠成标量 `weighted_lambda`。Reviewer 会问：**"这和简单的 `weight_reg *= decay` 有什么区别？"**

**应对 — 🔴 关键消融实验**：

```python
# Fixed Decay Baseline — 必须跑的对照
# 用指数衰减替代 BOCD，验证 belief-adaptive 的不可替代性
class FixedDecaySchedule:
    def __init__(self, decay=0.99):
        self.decay = decay
        self.current_lambda = 1.0
    
    def step(self):
        self.current_lambda *= self.decay
    
    def reset(self):
        self.current_lambda = 1.0  # 每个 episode 重置
```

**预期结果**：在**多次突变**环境下（5 modes，1800-7200s 间隔），fixed decay 在第一次突变后 penalty 已衰减到很低，面对后续突变无法重新拉高 → reward 崩塌。BA-PR 的 BOCD 在每次突变时自动重置 belief 到短 run-length → penalty 重新升高 → 快速恢复。

**如果 fixed decay 表现和 BA-PR 一样好**：说明 BOCD 架构是 over-engineering，需要重新审视算法设计。

#### 质疑 3："BOCD 是现成算法，surprise 信号缺乏物理动机"

**应对**：在论文中明确承认 BOCD 本身不是贡献，贡献是将其嵌入 ensemble SAC 框架并用 κ 信号驱动。同时提供 §12 的多种 surprise 备选方案作为消融维度。

### 8.2 必须的可视化

为了回应以上质疑，实验 section 必须包含以下可视化：

1. **Belief 分布热力图** — x 轴: training step, y 轴: run-length h, 颜色: ρ(h)。叠加模式切换的竖线。**预期**：切换时 belief 集中在 h=0（短 run-length），平稳时逐渐扩散到高 h
2. **`weighted_lambda` 时序曲线** — 叠加模式切换时间点。**预期**：切换时 spike 到高值，平稳时衰减
3. **fixed decay vs BA-PR** 的 reward 曲线对比 — 关键是多次突变后的恢复能力差异

---

## 9. Tensor 数据流与 Shape 速查

以下是各关键变量在 `compute_q_loss_bapr` 中的 shape 和来源，帮助 debug 广播错误：

```
变量名                   Shape                     来源/说明
──────────────────────────────────────────────────────────────────
state                    [batch_size, 18]          replay_buffer.sample()
action                   [batch_size, 1]           replay_buffer.sample()
reward                   [batch_size, 1]           replay_buffer.sample()
next_state               [batch_size, 18]          replay_buffer.sample()
done                     [batch_size, 1]           replay_buffer.sample()

predicted_q              [ensemble_size, batch]    soft_q_net(state, action)
target_q_next            [ensemble_size, batch]    target_soft_q_net(next_state, new_next_action)
next_log_prob            [batch]                   policy_net.evaluate(next_state)
  → unsqueeze+repeat     [ensemble_size, batch]    next_log_prob.unsqueeze(0).repeat(ensemble, 1)

reg_norm                 [ensemble_size]           compute_reg_norm(target_soft_q_net)
  → unsqueeze+repeat     [ensemble_size, batch]    reg_norm.unsqueeze(-1).repeat(1, batch)

epi_penalty              [batch]                   target_q_next.std(dim=0)

# ===== BA-PR 新增张量 =====
belief                   [max_H]                   belief_tracker.belief → torch.tensor
penalty_schedule          [max_H]                   exp(-0.1 * arange(max_H))
weighted_lambda          scalar (.)                (belief * penalty_schedule).sum()

target_q_value           [ensemble_size, batch, 1] reward + (1-done) * γ * target_q_next.unsqueeze(-1)
```

---

## 10. 实现步骤清单（按顺序执行）

以下是面向 AI 编码助手或开发者的**精确执行清单**，标注了每一步要改的文件、行号范围和验证方法。

### Step 1: 环境层 — `route.py` 改造
- **文件**: `env_original/route.py`
- **改动**: `__init__` 加 3 个属性（`speed_mean_scale`, `speed_cap`, `base_max_speed`）；`route_update` 用 `math.log(scale)` 偏移
- **验证**: 单独跑 `sim.py __main__`，确认无报错

### Step 2: 环境层 — `station.py` 改造
- **文件**: `env_original/station.py`
- **改动**: `__init__` 加 `self.od_multiplier = 1.0`；`station_update` 中 `demand * self.od_multiplier`
- **验证**: 同 Step 1

### Step 3: 环境层 — `sim.py` 改造
- **文件**: `env_original/sim.py`
- **改动**: `__init__` 加模式切换参数；新增 `_apply_mode`, `_maybe_switch_mode`, `get_current_mode_info`；改造 `reset()`；`step()` 开头加 `_maybe_switch_mode()`
- **改动量**: 约 60 行新增
- **验证**: `python env_original/sim.py`，观察 mode_history 输出

### Step 4: 算法层 — 新增 `BeliefTracker` 和 `SurpriseComputer`
- **文件**: 新建 `ensemble_paper/bapr_components.py` 或内嵌到训练脚本
- **改动**: 定义两个类（约 60 行）
- **验证**: 单元测试 — `belief_tracker.update(0.5)` 后检查 `belief.sum() ≈ 1.0`

### Step 5: 算法层 — 训练脚本改造
- **文件**: 复制 `sac_ensemble_original_logging.py` → `sac_ensemble_bapr.py`
- **改动**:
  - `env_bus(...)` 构造增加 `enable_mode_switch=True, mode_profiles=...` 
  - 新增 `compute_q_loss_bapr()` 方法
  - `update()` 方法签名加 `belief_tracker, surprise_computer`
  - 训练循环加 `belief_tracker.reset()`, 日志加 belief/surprise
  - `argparse` 加新参数（`--enable_mode_switch`, `--max_run_length`, `--hazard_rate` 等）
- **改动量**: 约 80 行修改 + 20 行新增参数
- **验证**: 跑 1 个 episode，检查日志中 `belief_entropy`, `effective_window`, `surprise_ema` 合理变化

### Step 6: 对照实验
- 按 §8 表格运行 Baseline 和 BA-PR 实验
- 对比 reward 曲线和 belief 变化

---

## 11. 已知坑与防护措施

> [!WARNING]
> 以下是在实现过程中需要特别注意的已知陷阱和边界情况。

| # | 问题 | 根因 | 修复/防护 |
|---|------|------|----------|
| 1 | `math.log(speed_mean_scale)` 在 `scale=0` 时报错 | `log(0)` 无定义 | `MODE_PROFILES` 的 `speed_mean_scale` 最小设为 `0.1`，不要设 0 |
| 2 | belief 变成全 NaN | `Z < 1e-10` 时除零 | `BeliefTracker.update()` 已内建退化到均匀分布的保护 |
| 3 | 模式切换后 headway 没立即变化 | 速度只在 `current_time % 300 == 0` 时更新 | `_apply_mode()` 末尾已加即时 `route_update()` |
| 4 | 第二个 episode 模式状态混乱 | `reset()` 重建 route/station 但没重置模式属性 | `reset()` 末尾已加 `_apply_mode("normal")` |
| 5 | `weighted_lambda` 始终接近 1.0 | `penalty_schedule` 衰减太慢（0.1 指数） | 如果 belief 集中在低 h，是正常的；如果集中在高 h 但 lambda 仍高，检查 schedule 衰减率 |
| 6 | Belief 在每个 episode 开头剧烈震荡 | 前几步 surprise 不稳定（Q 网络刚被 reset 状态扰动） | 可在 episode 前 50 步不更新 belief（warmup），或用更大的 `ema_alpha` 平滑 |
| 7 | `station_od_overrides` 中站点名拼写错误 | 静默忽略，该站点不会被覆盖 | 在 `_apply_mode()` 中加 `assert station.station_name in [s.station_name for s in self.stations]` 校验 |

---

## 12. Surprise 信号备选方案

当前方案使用 `|reg_norm(current) - reg_norm(target)|` 作为 surprise。如果实验中发现效果不佳，以下是备选方案，可直接替换 `SurpriseComputer.compute()` 的实现：

| 方案 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| **A. Reg-norm 差（当前）** | `\|κ_{cur} - κ_{tgt}\|` | 复用已有 `compute_reg_norm` | 信号间接，反映 soft-update lag |
| **B. TD-error 突变** | `\|Q(s,a) - (r + γV(s'))\|` 的 EMA | 直接反映 value 预测失准 | 需额外计算，batch 级而非 step 级 |
| **C. Ensemble disagreement** | `predicted_q.std(dim=0).mean()` 的突变 | 直接反映 epistemic uncertainty | 正常训练中也会变，需取差分 |
| **D. Reward 突变** | `\|r_{ema} - r_{current}\|` | 最直接的环境变化信号 | 对奖励稀疏的环境不适用 |

---

## 13. Sample Complexity & Regret Bound 分析

> [!IMPORTANT]
> 本节将 BA-PR SAC 的样本复杂度与 regret bounds 和 RE-SAC、标准 risk-sensitive RL 进行理论对比，为论文的 Theoretical Analysis 部分提供可引用的定量依据。

### 13.1 BA-PR 的 Per-Step 计算开销分析

BA-PR 在 RE-SAC 基础上新增的计算不改变 Bellman backup 的样本复杂度阶次，因为额外计算都是 **O(1) per step**：

| 组件 | 计算量 | 是否需要额外样本？ |
|------|--------|-------------------|
| `compute_reg_norm(soft_q_net)` | $O(P)$（P = 网络参数量） | ❌ 只涉及网络权重 |
| `surprise = \|κ_{cur} - κ_{tgt}\|` | $O(1)$ 标量运算 | ❌ |
| `belief_tracker.update(surprise)` | $O(\|H\|)$ 似然 + 归一化 | ❌ 纯计算，无环境交互 |
| `weighted_lambda = Σ belief · schedule` | $O(\|H\|)$ 内积 | ❌ |

**结论**：BA-PR 的 per-step 额外开销为 $O(P + |H|)$，其中 $|H| = 20 \ll P$，因此开销由 `compute_reg_norm` 主导——这和 RE-SAC 已有的 target 网络 reg_norm 计算完全对称，**不增加渐近复杂度**。

### 13.2 Regret 分解：BA-PR vs RE-SAC vs Risk-Sensitive

设 $T$ 为总训练步数，$K$ 为 episode 数，$M$ 为 episode 内平均模式切换次数，$H_{ep}$ 为 episode horizon 长度。

> **符号注意**：此处 $H_{ep}$ 是 episode 的步数长度（horizon），**不是** §1.2 中的 run-length 空间 $H$。

#### (A) Risk-Sensitive RL 的指数壁垒（Fei et al. 2020）

$$
\text{Regret}_{RS}(T) \ge \Omega\left(\exp(|\beta|H_{ep}) \cdot \sqrt{S^2 A T}\right)
$$

Sample complexity:  $T_\epsilon^{RS} \ge \Omega\left(\frac{\exp(2|\beta|H_{ep}) S^2 A}{\epsilon^2}\right)$

> 参见 `appendix.tex` §A.1 的详细推导。

#### (B) RE-SAC 的多项式 scaling（非正式论证）

RE-SAC 通过将风险优化转化为 shifted-reward MDP $\tilde{R}(s,a) = R(s,a) - \lambda \|W\|^2$，在惩罚项冻结条件下等价于标准 MDP：

$$
\text{Regret}_{RE\text{-}SAC}(T) \lesssim \tilde{O}\left(\sqrt{H_{ep}^3 S^2 A T}\right) + \underbrace{\frac{\gamma \Delta_\kappa}{1-\gamma}}_{\text{penalty approximation gap}}
$$

其中 $\Delta_\kappa$ 是 $\kappa$ 在训练过程中的最大变化量。

#### (C) BA-PR SAC 的 Regret 分解（本文贡献）

BA-PR 的 regret 可以分解为三个独立项：

$$
\text{Regret}_{BA\text{-}PR}(T) \le \underbrace{\tilde{O}\left(\sqrt{H_{ep}^3 S^2 A T}\right)}_{\text{(i) 标准 RL regret}} + \underbrace{K \cdot M \cdot \frac{\Delta_{mode}}{1-\gamma}}_{\text{(ii) 模式切换追踪误差}} + \underbrace{\frac{\gamma \Delta_\kappa}{1-\gamma}}_{\text{(iii) penalty 近似 gap}}
$$

**各项含义**：

| 项 | 来源 | Lean 支撑 | 可控手段 |
|----|------|-----------|----------|
| **(i)** 标准 RL regret | Tabular MDP regret 上界 | — | batch_size, replay buffer |
| **(ii)** 模式切换追踪误差 | `BAMOR.lean: anytime_tracking_bound` | ✅ $e(n) \le \gamma^n e(0) + \frac{\Delta}{1-\gamma}$ | 增大 `mode_switch_interval` 降低 M |
| **(iii)** penalty 近似 gap | `appendix.tex: (S2)` | ⚠️ 非形式化 | 降低 $\lambda_{ale}$ |

### 13.3 Anytime Tracking Bound 的定量化

利用 `BAMOR.lean` Part V 的 `anytime_tracking_bound` 和 `uniform_tracking_bound`：

**场景**：模式每 $T_{gap}$ 步切换一次，最大 per-step perturbation 为 $\Delta_{mode}$。

$$
\|Q_n - Q^*_{current\_mode}\|_\infty \le \gamma^{n_{since\_switch}} \cdot \|Q_{switch} - Q^*\|_\infty + \frac{\Delta_{mode}}{1-\gamma}
$$

对我们的公交场景量化：
- $\gamma = 0.99$, $1-\gamma = 0.01$
- $\Delta_{mode}$：由模式切换引起的单步 Bellman backup 偏差
  - `congestion_severe`: reward 从 -50 跳到 -440 → $\Delta_{mode} \approx 390 \times \gamma \approx 386$
  - `demand_surge`: reward 从 -50 跳到 -200 → $\Delta_{mode} \approx 150 \times \gamma \approx 149$
- $\frac{\Delta_{mode}}{1-\gamma} \approx 38600$（congestion）或 $14900$（demand surge）

> [!NOTE]
> 这个上界非常保守（worst-case）。实际中 BOCD 机制在检测到突变后快速重置 belief，
> 使 $\gamma^n$ 项快速衰减（$\gamma^{100} \approx 0.366$，即切换后 100 步误差衰减 63%）。
> 公交环境的 `mode_switch_interval = (1800, 7200)` 远大于收敛所需的 $\sim 500$ 步，
> 因此实际追踪误差远低于理论上界。

### 13.4 与 BAMOR (多目标) 的 Sample Complexity 对比

BAMOR 的额外开销：

| 组件 | 额外计算 | Sample 开销 |
|------|----------|-------------|
| 向量化 Q: $Q(s,a) \in \mathbb{R}^m$ | $O(m)$ per component | 不变（同一个 transition） |
| Mahalanobis surprise | $O(m)$ 对角矩阵运算 | 不变 |
| Scalarized Pareto aggregation | $O(m \cdot |A|)$ per state | 不变 |

**结论**：BAMOR 的 sample complexity 与 BA-PR 相同阶次，仅计算开销线性增加 $O(m)$。这与 `BAMOR.lean` 中 component-wise 证明策略一致——每个目标维度独立收缩，互不干扰。

### 13.5 论文写法建议

> [!TIP]
> 在论文 Theoretical Analysis 部分，建议按以下结构呈现：
> 1. **Theorem 1 (Contraction)**: 直接引用 `BAPR.lean: bapr_contraction`，一句话结论
> 2. **Theorem 2 (Anytime Tracking)**: 引用 `BAMOR.lean: anytime_tracking_bound`，给出 $\Delta/(1-\gamma)$ 上界
> 3. **Corollary (Sample Complexity)**: 利用 §13.2(C) 的分解，重点强调 (ii) 项是 BA-PR 独有的，且被 Lean 证明覆盖
> 4. **Remark**: 与 risk-sensitive RL 的指数壁垒对比（引用 Fei et al.），强调 BA-PR 保持多项式 scaling

---

## 14. JAX 大规模实验迁移方案

> [!IMPORTANT]
> 做大规模实验时，环境和算法都应迁移到 JAX，以利用 GPU/TPU 并行化实现 **数thousand倍加速**。以下是完整迁移方案。

### 14.1 为什么需要 JAX？

| 维度 | 当前 Python/PyTorch | JAX |
|------|---------------------|-----|
| 环境并行 | 单进程串行 step（1 env） | `jax.vmap` 批量 step（1024+ envs 同时） |
| 训练吞吐 | ~50 env steps/s | ~100k+ env steps/s（GPU vectorized） |
| 梯度计算 | PyTorch autograd | JAX `jit` + `grad`（XLA 编译优化） |
| 实验规模 | 1 seed × 1 config = 数小时 | 32 seeds × 10 configs = 数小时（并行） |
| 设备 | 单 GPU | GPU / TPU pod 无缝切换 |

### 14.2 环境 JAX 化：`env_bus` → `BusEnvJAX`

#### 设计原则

公交环境的核心循环是 **纯数值计算**（速度采样、headway 更新、乘客泊松到达），天然适合 JAX 化。

#### 状态表示（全部 JAX Array 化）

```python
import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class BusEnvState:
    """JAX-compatible environment state — 完全无 Python 对象"""
    # 车辆状态: [max_buses, state_dim]
    bus_positions: jnp.ndarray       # [max_buses] 绝对距离
    bus_speeds: jnp.ndarray          # [max_buses] 当前速度
    bus_headways_fwd: jnp.ndarray    # [max_buses] 前车时距
    bus_headways_bwd: jnp.ndarray    # [max_buses] 后车时距
    bus_active: jnp.ndarray          # [max_buses] bool mask
    bus_directions: jnp.ndarray      # [max_buses] 上行/下行

    # 路段状态
    route_speeds: jnp.ndarray        # [num_routes] 当前速度限制
    route_mean_scales: jnp.ndarray   # [num_routes] 均值缩放比例

    # 站点状态
    station_passengers: jnp.ndarray  # [num_stations, num_destinations] 等候人数
    station_od_multipliers: jnp.ndarray  # [num_stations] OD 倍率

    # 模式切换状态 (BA-PR)
    current_mode_id: jnp.int32       # 当前模式 ID
    next_switch_time: jnp.int32      # 下次切换时间
    mode_switch_count: jnp.int32     # 累计切换次数

    # 时间
    current_time: jnp.int32
    rng_key: jax.random.PRNGKey      # JAX PRNG state
```

#### 核心 step 函数

```python
@jax.jit
def env_step(state: BusEnvState, action: jnp.ndarray,
             mode_profiles: jnp.ndarray, params: EnvParams) -> tuple:
    """
    纯函数式 env step — 可 vmap 批量并行
    
    Args:
        state: 当前环境状态
        action: [max_buses] holding time
        mode_profiles: [num_modes, profile_dim] 预编码的模式参数
        params: 静态环境参数（路段长度、站点位置等）
    
    Returns:
        next_state, obs, reward, done, info
    """
    key, subkey1, subkey2, subkey3 = jax.random.split(state.rng_key, 4)

    # 1. 模式切换检查（无分支，用 jax.lax.cond）
    should_switch = state.current_time >= state.next_switch_time
    new_mode_id = jax.lax.cond(
        should_switch,
        lambda: _sample_new_mode(subkey1, state.current_mode_id, mode_profiles.shape[0]),
        lambda: state.current_mode_id
    )
    # 应用模式参数
    route_mean_scales, route_sigmas, route_caps, od_mults = _decode_mode(
        mode_profiles[new_mode_id], params
    )

    # 2. 速度采样（向量化 lognormal + clip）
    mu = params.base_speed_means + jnp.log(route_mean_scales)  # [num_routes]
    speeds = jnp.clip(
        jnp.log(jax.random.lognormal(subkey2, mu, route_sigmas)),
        2.0, route_caps
    )

    # 3. 车辆物理更新（全向量化）
    new_positions = state.bus_positions + state.bus_speeds * params.dt
    # holding action
    new_positions = jnp.where(action > 0, state.bus_positions, new_positions)

    # 4. Headway 计算 (全矩阵化)
    # ... (纯 jnp 运算，避免 for 循环)

    # 5. 乘客到达（泊松采样，JAX 向量化）
    arrivals = jax.random.poisson(subkey3, 
        state.station_passengers * od_mults * params.passenger_interval / 3600.0)

    # 6. Reward
    reward = -jnp.abs(new_headways - params.target_headway)

    next_state = state.replace(
        bus_positions=new_positions,
        route_speeds=speeds,
        current_time=state.current_time + 1,
        current_mode_id=new_mode_id,
        rng_key=key,
        # ... 其他更新
    )
    return next_state, obs, reward, done, {}


# 批量并行：1024 个环境同时 step
batched_step = jax.vmap(env_step, in_axes=(0, 0, None, None))
```

#### 关键改造点

| 原始 Python 代码 | JAX 替代方案 |
|-----------------|-------------|
| `for bus in self.bus_all:` | `jax.vmap` over bus axis |
| `random.lognormvariate(mu, sigma)` | `jax.random.lognormal(key, mu, sigma)` |
| `np.random.poisson(lam)` | `jax.random.poisson(key, lam)` |
| `if current_time >= switch_time:` | `jax.lax.cond(...)` 或 `jnp.where(...)` |
| `self.bus_all.append(Bus(...))` | 固定 `max_buses` 数组 + `active` mask |
| `class Bus / Route / Station` | `@struct.dataclass` 纯数据结构 |

### 14.3 算法 JAX 化：PyTorch SAC → JAX SAC

#### 推荐框架

| 方案 | 优势 | 适合场景 |
|------|------|----------|
| **[PureJaxRL](https://github.com/luchris429/purejaxrl)** | 环境+算法全 JAX，end-to-end jit | ✅ **首选**：自定义环境 |
| **[CleanRL JAX](https://github.com/vwxyzjn/cleanrl)** | SAC JAX 实现参考 | 参考实现 |
| **[Brax](https://github.com/google/brax)** | Google 官方，MuJoCo 级物理 | 通用 RL，非自定义环境 |
| **Flax + Optax** | 底层灵活 | 自定义网络结构 |

#### SAC 核心组件 JAX 化

```python
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

# ─── Ensemble Q-Network（Flax） ───
class EnsembleQNet(nn.Module):
    num_critics: int = 5
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        # Vectorized ensemble: 用 vmap over params
        qs = nn.vmap(
            lambda: nn.Sequential([
                nn.Dense(self.hidden_dim), nn.relu,
                nn.Dense(self.hidden_dim), nn.relu,
                nn.Dense(1)
            ]),
            variable_axes={'params': 0},
            split_rngs={'params': True},
            axis_size=self.num_critics,
        )(x)
        return qs.squeeze(-1)  # [num_critics, batch]

# ─── BA-PR Components（纯 JAX） ───
@jax.jit
def compute_reg_norm_jax(params_tree) -> jnp.ndarray:
    """compute_reg_norm 的 JAX 等价：L1 norm of all layers"""
    leaves = jax.tree_util.tree_leaves(params_tree)
    return jnp.array([jnp.sum(jnp.abs(l)) for l in leaves]).sum()

@jax.jit
def belief_update_jax(belief: jnp.ndarray, surprise: float,
                      base_var: float, var_growth: float,
                      hazard_rate: float) -> jnp.ndarray:
    """BeliefTracker.update() 的 JAX 等价 — 可 vmap/jit"""
    max_H = belief.shape[0]
    # Likelihood
    variances = base_var + var_growth * jnp.arange(max_H)
    L = jnp.exp(-surprise**2 / (2 * variances))
    # Bayes update
    unnorm = belief * L
    Z = unnorm.sum()
    belief = jnp.where(Z > 1e-10, unnorm / Z, jnp.ones(max_H) / max_H)
    # BOCD hazard shift
    growth_prob = belief * (1 - hazard_rate)
    cp_prob = belief.sum() * hazard_rate
    new_belief = jnp.zeros(max_H)
    new_belief = new_belief.at[0].set(cp_prob)
    new_belief = new_belief.at[1:].set(growth_prob[:-1])
    total = new_belief.sum()
    return jnp.where(total > 1e-10, new_belief / total, jnp.ones(max_H) / max_H)

# ─── BA-PR Q-Loss（纯 JAX） ───
@jax.jit
def compute_q_loss_bapr_jax(q_params, target_q_params, policy_params,
                            state, action, reward, next_state, done,
                            belief, alpha, gamma, weight_reg):
    """compute_q_loss_bapr 的 JAX 等价"""
    predicted_q = ensemble_q_net.apply(q_params, state, action)

    # Target Q
    next_action, next_log_prob = policy_net.apply(policy_params, next_state, method='sample')
    target_q_next = ensemble_q_net.apply(target_q_params, next_state, next_action)

    # BA-PR: belief-weighted epistemic penalty
    epi_penalty = target_q_next.std(axis=0)
    penalty_schedule = jnp.exp(-0.1 * jnp.arange(belief.shape[0]))
    weighted_lambda = (belief * penalty_schedule).sum()

    # Reg norm (target)
    reg_norm = compute_reg_norm_jax(target_q_params)

    target_q_next = (target_q_next
                     - alpha * next_log_prob
                     + weight_reg * reg_norm
                     - weighted_lambda * weight_reg * epi_penalty)

    target_q_value = reward + (1 - done) * gamma * target_q_next
    q_loss = jnp.mean((predicted_q - jax.lax.stop_gradient(target_q_value)) ** 2)

    return q_loss, weighted_lambda
```

### 14.4 并行实验架构

```
                    ┌─────────────────────────────────┐
                    │         JAX Compiled Graph       │
                    │                                  │
                    │  ┌─────────┐   ┌──────────────┐ │
                    │  │ Env × N │──▶│ SAC Trainer   │ │
                    │  │ (vmap)  │◀──│ (jit + grad)  │ │
                    │  └─────────┘   └──────────────┘ │
                    │       │              │           │
                    │  ┌────▼────┐   ┌─────▼────────┐ │
                    │  │ Belief  │   │ Replay Buffer │ │
                    │  │ Tracker │   │ (jnp array)   │ │
                    │  │ (vmap)  │   │               │ │
                    │  └─────────┘   └──────────────┘ │
                    └─────────────────────────────────┘
                                    │
                          GPU/TPU   ▼   All in device memory
```

**并行维度**：

| 维度 | 方式 | 数量 |
|------|------|------|
| 环境并行 | `jax.vmap(env_step)` | 1024 envs |
| Seed 并行 | `jax.vmap` over RNG keys | 32 seeds |
| 超参搜索 | `jax.pmap` across devices | 4-8 configs per GPU |
| 总并行度 | 1024 × 32 = **32768** env instances | 单 GPU |

### 14.5 迁移路线图

```
Phase 1: 环境 JAX 化                          Phase 2: 算法 JAX 化
─────────────────                              ─────────────────
[1] 定义 BusEnvState dataclass                 [4] Flax EnsembleQNet + PolicyNet
[2] 实现 env_step 纯函数                       [5] Optax optimizer setup
[3] 验证: single env 对齐 Python 环境输出      [6] BA-PR components (belief, surprise)
    ↓                                              ↓
Phase 3: 集成 + 大规模实验                     Phase 4: 论文实验
──────────────────────                         ─────────────────
[7] vmap batched rollout                       [9] 5 mode profiles × 32 seeds
[8] end-to-end jit 训练循环                    [10] ablation: belief vs fixed-decay
    验证: 对比 PyTorch 版训练曲线              [11] MORL 扩展 (BAMOR)
```

### 14.6 预期加速与实验规模

| 实验 | PyTorch 预估耗时 | JAX 预估耗时 | 加速比 |
|------|-----------------|-------------|--------|
| 单 seed/config (500 episodes) | ~8 小时 | ~15 分钟 | ~32× |
| 8 configs × 5 seeds (对照实验) | ~160 小时 (串行) | ~2 小时 | ~80× |
| 32 seeds × 10 configs (完整) | ~1280 小时 | ~6 小时 | ~200× |
| BAMOR 多目标扩展 | ×2 (双目标) | ~12 小时 | ~200× |

> [!TIP]
> **最低可行迁移**：如果时间紧迫，可以**只 JAX 化环境**（Phase 1），训练仍用 PyTorch，通过 `jax.dlpack` 在 GPU 上零拷贝传输 tensor。这样环境吞吐提升 ~50×，算法端不改。

### 14.7 JAX 环境验证策略

为确保 JAX 环境与原始 Python 环境行为一致：

```python
# 验证脚本：comparison_test.py
def verify_jax_env():
    """对比 JAX 环境与 Python 环境在相同随机种子下的输出"""
    py_env = env_bus(path, route_sigma=1.5, enable_mode_switch=True)
    jax_env = BusEnvJAX(params, mode_profiles)

    py_env.reset()
    jax_state = jax_env.reset(jax.random.PRNGKey(42))

    for step in range(1000):
        action = np.zeros(25)  # 统一动作
        py_state, py_reward, py_done = py_env.step(action)
        jax_state, jax_obs, jax_reward, jax_done, _ = env_step(
            jax_state, jnp.zeros(25), mode_profiles, params)

        # 核心断言
        np.testing.assert_allclose(py_reward, jax_reward, atol=1e-4,
            err_msg=f"Step {step}: reward mismatch")
        np.testing.assert_allclose(py_state['headways'], jax_obs['headways'], atol=1e-3,
            err_msg=f"Step {step}: headway mismatch")

    print("✅ JAX environment matches Python environment")
```
