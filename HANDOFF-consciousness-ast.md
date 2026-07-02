# 衔接文档：意识理论 × mycc 架构升级（优先级 1-3）

> Phase 1-5 已全部完成并推送（commit `a03b1cc`）。本文档用于优先级 1-3 接力。

## 已完成（Phase 1-5 + 推送）

### Phase 1: 检索轨迹反馈闭环 ✅
- `memory-db.ts` — `aggregateTraces()` + `TraceStats` 接口（聚合 7 天轨迹）
- `context-loader.ts` — `applyHistoricalBoost()`（RRF 后叠加历史命中率 0.2x，10min 缓存）
- `attention-audit` skill — SKILL.md 已创建（每日 23:00 检索审计）

### Phase 2: 注意力预测 ✅
- `BRO-PDCA方法论.md` — P1 阶段新增「注意力预测声明」
- `context-loader.ts` — `RetrievalTrace` 接口新增 `predicted?: string[]` 字段

### 更早完成的（P0 + P1）
- CLAUDE.md: HOT 置信度自评 + RPT 持续验证循环
- BRO-PDCA: PP 预测-修正循环（预期结果声明+惊讶度量表）
- memory-system.md: IIT v6 因果关联标记规范
- context-loader.ts: `resolveCausalChains()` IIT 因果链扩展召回
- status-updater.ts: `loadAttentionSchema()` AST 注意力图式
- blackboard.md: GWT 三引擎信息黑板
- multi-agent/RULES.md: 信息黑板 section
- 认知沉淀: `3-Thinking/06-意识理论与认知架构映射.md`

---

## 已完成（2026-03-31 意识架构优先级 1-4 + 路由分发）

### P1 PP 自动验证闭环 ✅
- `memory-db.ts`: traces 表 predicted 列 + migration + `verifyPredictions()` + `PredictionVerification` 接口
- `context-loader.ts`: `parsePredictedCategories()` + `getVerificationReport()` 导出
- 数据流：prompt `[注意力预测]` → saveTrace(predicted) → attention-audit 调 verifyPredictions

### P2 AST 路由信任度 ✅
- `task-queue.ts`: `computeEngineTrust(cwd)` 解析 task-log.md + 10min 缓存
- `autoRouteEngine(msg, cwd)` 叠加信任度，reason 附 `[信任度: xx%]`
- codex 低信任 + gemini 明显更高 → 降级到 gemini

### 路由分发集成 ✅
- `http-server.ts`: autoRouteEngine 传入 cwd（信任度生效）
- chat→task 升级：chat 消息如果路由到非 cc，自动升级为 task 走异步路由
- 调用链完整：autoRouteEngine → taskQueue → processCliEngineMessage → runCliEngine

### P3 GWT 并行竞争 — 方案已确认，待实施
- 前置依赖（路由分发）已完成
- 约束：竞争前飞书确认老板、胜负标准可迭代自学习、每小时≤3次
- 详见 evolution-log 2026-03-31 记录

### P4 IIT 因果关联 — 挂起
- 等 multi-agent 真正产生跨引擎数据后再激活

### Phase 4: 进化策略元修改（原 HANDOFF 待完成）
**关键文件**：
- `0-System/evolution-log.md` — 增加「进化策略评估（Meta-Evolution）」section
- `.claude/skills/metaclaw/SKILL.md` — 追加第 7 层：元策略评估

### Phase 5: 认知镜子（原 HANDOFF 待完成）
**关键文件**：
- `context-loader.ts` — 计算注意力负载 + 知识边界感知
- `status-updater.ts` — 自动写入 blackboard 注意力状态
- `blackboard.md` — 格式扩展（负载、知识薄弱区）

---

## 衔接提示词

新 session 粘贴以下内容即可续上：

```
继续 AST 注意力图式升级 Phase 3-5。

上下文：
1. 读 `3-Thinking/06-意识理论与认知架构映射.md` 了解整体框架
2. 读 `.claude/plans/witty-hatching-glacier.md` 了解完整计划
3. 读 `2-Projects/cc-eye/HANDOFF-consciousness-ast.md` 了解已完成/待完成

Phase 1-2 已完成（检索轨迹反馈闭环 + 注意力预测）。

现在做 Phase 3（社会注意力）：
- 扩展 task-queue.ts 的 autoRouteEngine() 返回 contextHints
- 在 http-server.ts 的 processFeishuMessage() 集成路由分发
- 实现 blackboard.md 的代码读写

然后 Phase 4（元策略）和 Phase 5（认知镜子）。

注意：Phase 3 涉及 http-server 改动，完成后需要重启后端验证。
```
