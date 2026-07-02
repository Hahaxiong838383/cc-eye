# 衔接文档：认知架构升级 v2（意识理论 + 前沿评价体系）

> 2026-03-31 session 产出。本文档覆盖意识理论 P1-P4 + 前沿评价体系（EDM/FadeMem/GWT）全部改动。

## 已完成

### 意识理论优先级 1-4

| 优先级 | 内容 | 文件 | 状态 |
|--------|------|------|------|
| P1 PP 自动验证闭环 | `verifyPredictions()` + `parsePredictedCategories()` + `getVerificationReport()` | memory-db.ts + context-loader.ts | ✅ |
| P2 AST 路由信任度 | `computeEngineTrust()` 解析 task-log.md + `autoRouteEngine(msg, cwd)` 叠加信任度 | task-queue.ts | ✅ |
| P3 GWT 并行竞争 | 方案确认，约束记录在 evolution-log | — | 待实施 |
| P4 IIT 因果关联 | 挂起，等 multi-agent 跑起来 | evolution-log | 挂起 |

### 路由分发集成

| 改动 | 文件 | 说明 |
|------|------|------|
| cwd 传参 | http-server.ts L520 | `autoRouteEngine(msg, this.cwd)` — 信任度权重生效 |
| chat→task 升级 | http-server.ts L513 | chat 消息如果路由到非 cc，自动升级走异步路由 |

### 前沿评价体系对标改进

| # | 改进 | 对标方案 | 文件 | 状态 |
|---|------|---------|------|------|
| 1 | EDM 记忆准入拦截层 | Eval-Driven Memory | memory-db.ts（items 表 quality_score 列 + `computeQualityScore()`) | ✅ 数据写入 |
| 2 | FadeMem 访问频率衰减 | FadeMem 指数衰减 | memory-db.ts（last_accessed/access_count 列 + `recordItemAccess()` + `getQualityAndDecayScores()`）+ context-loader.ts（注入时记录访问） | ✅ 数据写入 |
| 6 | Δ 自动巩固 | PI-LLM 睡眠巩固 | docs/memory-system.md（遗忘规则增加 Δ ≥ 3 自动重建基线） | ✅ 规则定义 |
| 8 | 质量分激活到检索评分 | EDM + FadeMem | `scoreSectionItems` 叠加 quality × decayFactor | ⏰ 定时任务 4/4 10:00 自动触发 |

---

## 待完成（按优先级）

### 近期（1-2 周）

| # | 改进 | 说明 | 预估工作量 | 状态 |
|---|------|------|-----------|------|
| 8 | **质量分激活**（4/4 自动触发） | scheduled-task `activate-quality-scores` 已创建，4/4 10:00 自动跑 | 半小时 | ⏰ 等 4/4 |
| 5 | **行为漂移量化监控** | memory-db.ts `computeBehaviorDrift()` + attention-audit skill v2（双轴：注意力+行为漂移，5项指标） | — | ✅ 2026-03-31 |
| 3 | **GWT 并行竞争** | **阻塞**：task-log 100% cc 引擎，codex/gemini 路由无实战数据。需先手动验证路由端到端可用性 | 半天 | ⏸️ 等路由数据 |

### 中期（1 个月）

| # | 改进 | 说明 | 状态 |
|---|------|------|------|
| 4 | **独立验证通道** | cc 生成 → codex/gemini 验证，分离 generator/verifier。等路由分发稳定后实施 | 挂起 |
| — | **Phase 4 元策略评估** | evolution-log Meta-Evolution 首次评估完成（2026-03-31）。淘汰条件收紧到 21 天；路由准确度维度数据不足 | ✅ 首次评估 |
| — | **Phase 5 认知镜子** | context-loader 注意力负载计算 + status-updater 自动写入 blackboard | 挂起 |

### 长期 / 观望

| # | 改进 | 说明 |
|---|------|------|
| 7 | 惊讶度事件分割（EM-LLM） | 需要本地 LLM 算 perplexity，M5 Air 无独显，成本高 |
| — | IIT 因果关联激活 | 等 multi-agent 真正产生跨引擎数据 |

---

## 关键文件索引

| 文件 | 改了什么 |
|------|---------|
| `.claude/skills/mycc/scripts/src/services/memory-db.ts` | items 表 3 新列 + migration + `computeQualityScore` + `recordItemAccess` + `getQualityAndDecayScores` + `verifyPredictions` + `PredictionVerification` + traces 表 predicted 列 |
| `.claude/skills/mycc/scripts/src/services/context-loader.ts` | `parsePredictedCategories` + `getVerificationReport` + saveTrace 传 predicted + 两处 recordItemAccessDB + import 更新 |
| `.claude/skills/mycc/scripts/src/task-queue.ts` | `computeEngineTrust(cwd)` + `autoRouteEngine(msg, cwd)` 信任度叠加 + import fs/path |
| `.claude/skills/mycc/scripts/src/http-server.ts` | autoRouteEngine 传 cwd + chat→task 路由升级 |
| `docs/memory-system.md` | 遗忘规则新增 Δ 自动巩固 |
| `0-System/evolution-log.md` | P1-P4 记录 + P3 GWT 约束 + Meta-Evolution 首次评估（2026-03-31） |
| `2-Projects/cc-eye/HANDOFF-consciousness-ast.md` | 状态更新（已完成/待完成） |

---

## 衔接提示词

新 session 粘贴以下内容续上：

```
继续认知架构优化。

上下文：读 `2-Projects/cc-eye/HANDOFF-cognitive-arch-v2.md`

当前状态：
- EDM + FadeMem 数据管道已铺好，4/4 scheduled-task 会自动激活质量分到检索评分
- 路由分发已集成（autoRouteEngine + chat→task 升级），等数据积累
- GWT 竞争方案已确认（evolution-log），等路由稳定后实施

优先做：
1. #5 行为漂移监控 — 改 attention-audit skill 追加风格指标
2. #3 GWT 并行竞争 — 需确认路由数据是否足够
3. Phase 4 元策略评估 — evolution-log + metaclaw 第 7 层

注意：如果 4/4 的 scheduled-task 没有自动跑，手动执行 `activate-quality-scores` 任务。
```
