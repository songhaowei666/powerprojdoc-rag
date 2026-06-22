# 局部 Spec 修订（Amendments）

本目录存放**尚未合并**或**已合并**进主 spec 的局部变更说明。主 spec 仍为 `spec/<module>_spec.md` 的唯一权威来源；合并后须同步更新主 spec。

## 目录结构

```
spec/amendments/
  README.md
  TEMPLATE.md
  pending/<module>/     # 进行中，待合并
  merged/<module>/      # 已合并进主 spec 的历史记录
```

- `<module>` 与主 spec 文件名一致（不含 `_spec.md`），例如 `generage_graph` → `spec/generage_graph_spec.md`。
- 首次为某模块写 amendment 时，在 `pending/` 或 `merged/` 下新建对应子目录。

## 文件命名

按时间自然排序：

```
YYYY-MM-DD_NN_<slug>.md
```

| 部分 | 说明 |
|------|------|
| `YYYY-MM-DD` | 创建或定稿日期 |
| `NN` | 同日内序号，从 `01` 起 |
| `slug` | 简短英文，描述改动点（小写、连字符） |

示例：`pending/generage_graph/2026-06-22_01_route-after-grade.md`

## 工作流

1. 在 `pending/<module>/` 新建 amendment（可复制 `TEMPLATE.md`）。
2. 实现代码时以 amendment + 主 spec 为准；合并前主 spec 可暂不改动。
3. 合并进 `spec/<module>_spec.md` 后：
   - 将 amendment 文件移至 `merged/<module>/`（保持原文件名）；
   - 在 amendment 顶部将 `Status` 改为 `merged`，并填写 `Merged at`。
4. 禁止在 `pending/` 与 `merged/` 各留一份重复文件。

## 与 AGENTS.md 的关系

- 小范围、可独立评审的改动：可先写 amendment，再实现、再合并主 spec。
- 已合并行为：以主 spec 为准；`merged/` 仅作历史追溯。
