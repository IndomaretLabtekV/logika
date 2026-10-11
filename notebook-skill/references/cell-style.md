# Notebook Cell Style Rules

These rules are notebook-specific. General prose rules (em dashes, colons, semicolons, non-ASCII, voice) are covered by the `/writing` skill, which is invoked at the start of every notebook session.

## Banner Cells

Three banner cells open every notebook. The first shows project name + tagline, the second shows team name/group + member list, and the third shows the TOC. Each starts with `---`. Do not merge them into one cell.

## Imports

Never place `import` statements inside any cell other than the Libraries cell. This includes inside function definitions, `try` blocks, or any other cell. If a fix requires a new import, add it to the Libraries cell.

## Docstrings

No docstrings in any notebook code cell. The bridging paragraph before the cell already explains what the code does.

## Divider Comments

Do not write `# --- label ---` or any comment whose sole purpose is visual separation inside a code cell string. Blank lines do the grouping. Section divider comments are fine in the builder `.py` file itself, but they must not appear in the string content passed to `code()`.

## Formula and Derivative Comments

Formula comments go **above** the code line they annotate, not trailing inline. The exception is tightly coupled single-expression index calculations where the comment explains that exact expression.

```python
# Right: formula comment above
# dL/dW = x^T @ dL/dy
self.grad_kernel = x_2d.T @ g_2d

# Right: index arithmetic inline
self._argmax[:, i, j, :, 0] = i * sH + idx // pW  # row = patch_row + flat_idx // pW

# Wrong: formula comment trailing on a general assignment
self.grad_kernel = x_2d.T @ g_2d  # dL/dW = x^T @ dL/dy
```

## Bridging Paragraphs

Immediately after `section()` and after every `## Subsection` header, add a markdown cell with a short prose paragraph that sets context before the first code cell. The bridging paragraph answers "what is happening here and why" and is not a bullet list. Write in third person.

```python
cells.append(section("Data Loading", "1"))
cells.append(md(
    "The class distribution and a sample image grid are visualised to verify the "
    "loading pipeline before training. All image loading uses the PIL-based "
    "`load_image` utility with no framework preprocessing layers involved."
))
```

## Insights Blocks

End each analysis sub-section with a `#### Insights` heading followed by a `>` blockquote, both in a single `md()` call. When an insight covers more than one distinct observation, split it into multiple blockquote paragraphs within the same cell.

```python
# Short insight
cells.append(md(
    "#### Insights\n\n"
    "> The dataset is nearly balanced across all six classes, "
    "so macro F1-score is a fair summary metric."
))

# Long insight split across paragraphs
cells.append(md(
    "#### Insights\n\n"
    "> The dataset is nearly balanced across all six classes in the train split, "
    "with each class contributing roughly 14-17% of the total. Because the distribution "
    "is close to uniform, macro F1-score is a fair summary metric and no class weighting "
    "is required.\n\n"
    "> The test split follows the same pattern, which means evaluation scores are "
    "directly comparable across classes without reweighting."
))
```
