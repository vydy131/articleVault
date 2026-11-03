
```dataview
TABLE WITHOUT ID
file.link AS "Недавно измененные",
file.mtime AS "Время изменения"
FROM -"template"
SORT file.mtime DESC
LIMIT 8
```
---

```dataview
TABLE WITHOUT ID 
out AS "Ссылки на интересное", 
rows.file.link AS "Откуда ноги растут"
FLATTEN file.outlinks AS out 
WHERE !(out.file) AND !contains(meta(out).path, "/") 
GROUP BY out 
SORT length(rows.file.link) DESC
```

--- 
```dataview
TABLE WITHOUT ID
  file.link AS "Карта контента",
  (length(file.inlinks) + length(file.outlinks)) AS "Количество связей"
FROM -"template"
WHERE contains(tags, "type/article")
SORT (length(file.inlinks) + length(file.outlinks)) DESC
```

---
```dataview
TABLE WITHOUT ID
file.link as "Сироты", file.path as "Путь"
FROM "base"
WHERE (!(file.inlinks) AND !(file.outlinks))
```
