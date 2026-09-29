# 自测指南

```bash
BASE=http://<服务器>:8002
KEY=<MEMORY_SYSTEM_KEY>

# 1) 健康
curl -s $BASE/health

# 2) 文本 Add
curl -s -X POST $BASE/v1/memory/add \
  -H "Authorization: Token $KEY" -H "Content-Type: application/json" \
  -d '{"request_id":"t1","user_id":"u1","session_id":"s1","messages":[{"role":"user","content":"我喜欢在落基山徒步，每年夏天都去。"},{"role":"assistant","content":"听起来很棒！你通常和谁一起去？"}]}'

# 3) 带图 Add（base64 data URI）
IMG=$(base64 -w0 test.jpg)
curl -s -X POST $BASE/v1/memory/add \
  -H "Authorization: Token $KEY" -H "Content-Type: application/json" \
  -d "{\"request_id\":\"t2\",\"user_id\":\"u1\",\"session_id\":\"s1\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"看看这张照片\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$IMG\"}}]}]}"

# 4) Search
curl -s -X POST $BASE/v1/memory/search \
  -H "Authorization: Token $KEY" -H "Content-Type: application/json" \
  -d '{"query":"用户喜欢在哪里徒步？","top_k":100}'

# 5) 幂等验证：重复 request_id 返回 duplicate:true
curl -s -X POST $BASE/v1/memory/add ... -d '{"request_id":"t1", ...}'
```
