from fastapi import Header, HTTPException

API_KEY_TO_TENANT = {
    "alpha-key-00000000000000000000000000000001": "tenant_alpha",
    "beta-key-00000000000000000000000000000001": "tenant_beta",
    "test-key-00000000000000000000000000000001": "tenant_test",
}


def get_tenant_id(api_key: str) -> str | None:
    return API_KEY_TO_TENANT.get(api_key)


def require_api_key(x_api_key: str = Header(...)) -> str:
    tenant_id = get_tenant_id(x_api_key)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return tenant_id
