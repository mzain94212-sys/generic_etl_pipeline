"""
Local Elasticsearch REST Compatible Server.
Provides native REST endpoints on port 9200 compatible with Elasticsearch 8.x/9.x.
"""

import json
import time
from typing import Any, Dict
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
import uvicorn

# In-memory document storage
INDEX_STORE: Dict[str, Dict[str, Any]] = {}


async def root(request):
    """Elasticsearch root endpoint GET /."""
    return JSONResponse({
        "name": "generic-etl-node-1",
        "cluster_name": "generic-elasticsearch-cluster",
        "cluster_uuid": "eTL-989c8349-ef4f-404d-927e",
        "version": {
            "number": "8.14.0",
            "build_flavor": "default",
            "build_type": "standalone",
            "build_hash": "219f21b33367123",
            "build_date": "2026-08-27T00:00:00.000Z",
            "build_snapshot": False,
            "lucene_version": "9.10.0",
            "minimum_wire_compatibility_version": "7.17.0",
            "minimum_index_compatibility_version": "7.0.0"
        },
        "tagline": "You Know, for Search"
    })


async def cluster_health(request):
    """Cluster health endpoint GET /_cluster/health."""
    return JSONResponse({
        "cluster_name": "generic-elasticsearch-cluster",
        "status": "green",
        "timed_out": False,
        "number_of_nodes": 1,
        "number_of_data_nodes": 1,
        "active_primary_shards": len(INDEX_STORE),
        "active_shards": len(INDEX_STORE),
        "relocating_shards": 0,
        "initializing_shards": 0,
        "unassigned_shards": 0
    })


async def cat_indices(request):
    """Cat indices endpoint GET /_cat/indices."""
    lines = ["health status index uuid pri rep docs.count docs.deleted store.size pri.store.size"]
    for idx_name, idx_data in INDEX_STORE.items():
        doc_count = len(idx_data.get('docs', {}))
        lines.append(f"green open {idx_name} {idx_name}-uuid 1 0 {doc_count} 0 {doc_count * 1.5:.1f}kb {doc_count * 1.5:.1f}kb")
    return Response("\n".join(lines), media_type="text/plain")


async def handle_index(request):
    """Handle index operations (GET/PUT/DELETE /<index>)."""
    index_name = request.path_params['index'].lower()
    method = request.method

    if method == "GET":
        if index_name in INDEX_STORE:
            return JSONResponse({index_name: INDEX_STORE[index_name].get('mapping', {})})
        return JSONResponse({"error": f"index_not_found_exception: {index_name}"}, status_code=404)

    elif method == "PUT":
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        INDEX_STORE[index_name] = {
            'mapping': body,
            'docs': {},
            'created_at': time.time()
        }
        return JSONResponse({"acknowledged": True, "shards_acknowledged": True, "index": index_name})

    elif method == "DELETE":
        if index_name in INDEX_STORE:
            del INDEX_STORE[index_name]
            return JSONResponse({"acknowledged": True})
        return JSONResponse({"acknowledged": False}, status_code=404)

    elif method == "HEAD":
        status = 200 if index_name in INDEX_STORE else 404
        return Response(status_code=status)


async def handle_mapping(request):
    """GET /<index>/_mapping."""
    index_name = request.path_params['index'].lower()
    if index_name in INDEX_STORE:
        return JSONResponse({index_name: {"mappings": INDEX_STORE[index_name].get('mapping', {}).get('mappings', {})}})
    return JSONResponse({"error": f"index_not_found_exception: {index_name}"}, status_code=404)


async def handle_bulk(request):
    """POST /_bulk or POST /<index>/_bulk."""
    raw_body = await request.body()
    text = raw_body.decode('utf-8', errors='ignore')
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    items = []
    i = 0
    while i < len(lines):
        try:
            action_meta = json.loads(lines[i])
            action_type = list(action_meta.keys())[0]
            meta_info = action_meta[action_type]
            index_name = meta_info.get('_index', request.path_params.get('index', 'generic_dataset')).lower()
            doc_id = str(meta_info.get('_id', i // 2 + 1))

            if index_name not in INDEX_STORE:
                INDEX_STORE[index_name] = {'mapping': {}, 'docs': {}, 'created_at': time.time()}

            if action_type in ['index', 'create']:
                if i + 1 < len(lines):
                    doc_body = json.loads(lines[i + 1])
                    INDEX_STORE[index_name]['docs'][doc_id] = doc_body
                    items.append({action_type: {"_index": index_name, "_id": doc_id, "_version": 1, "result": "created", "status": 201}})
                    i += 2
                    continue
            i += 1
        except Exception:
            i += 1

    return JSONResponse({
        "took": 1,
        "errors": False,
        "items": items
    })


async def handle_search(request):
    """GET/POST /<index>/_search or /_search."""
    index_name = request.path_params.get('index', '').lower()
    query_param = request.query_params.get('q', '')
    
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    results = []
    indices_to_search = [index_name] if index_name and index_name in INDEX_STORE else list(INDEX_STORE.keys())

    search_term = query_param.lower()
    if not search_term and 'query' in body:
        q = body['query']
        if 'multi_match' in q:
            search_term = str(q['multi_match'].get('query', '')).lower()
        elif 'match' in q:
            f_key = list(q['match'].keys())[0]
            search_term = str(q['match'][f_key]).lower()
        elif 'query_string' in q:
            search_term = str(q['query_string'].get('query', '')).lower()

    for idx in indices_to_search:
        for doc_id, doc in INDEX_STORE[idx]['docs'].items():
            if not search_term or search_term == "*":
                results.append({"_index": idx, "_id": doc_id, "_score": 1.0, "_source": doc})
            else:
                for k, v in doc.items():
                    if search_term in str(v).lower():
                        results.append({"_index": idx, "_id": doc_id, "_score": 1.0, "_source": doc})
                        break

    return JSONResponse({
        "took": 1,
        "timed_out": False,
        "_shards": {"total": len(indices_to_search), "successful": len(indices_to_search), "skipped": 0, "failed": 0},
        "hits": {
            "total": {"value": len(results), "relation": "eq"},
            "max_score": 1.0,
            "hits": results
        }
    })


app = Starlette(
    debug=True,
    routes=[
        Route("/", root, methods=["GET"]),
        Route("/_cluster/health", cluster_health, methods=["GET"]),
        Route("/_cat/indices", cat_indices, methods=["GET"]),
        Route("/_bulk", handle_bulk, methods=["POST"]),
        Route("/_search", handle_search, methods=["GET", "POST"]),
        Route("/{index}/_mapping", handle_mapping, methods=["GET"]),
        Route("/{index}/_bulk", handle_bulk, methods=["POST"]),
        Route("/{index}/_search", handle_search, methods=["GET", "POST"]),
        Route("/{index}", handle_index, methods=["GET", "PUT", "DELETE", "HEAD"]),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9200, log_level="warning")
