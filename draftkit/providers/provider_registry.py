from draftkit.providers.provider_base import MockSportsbookProvider


_PROVIDERS = {}
_ACTIVE_PROVIDER_NAME = None


def register_provider(provider, make_active=False):
    provider_name = provider.get_provider_name()
    _PROVIDERS[provider_name] = provider

    global _ACTIVE_PROVIDER_NAME
    if make_active or _ACTIVE_PROVIDER_NAME is None:
        _ACTIVE_PROVIDER_NAME = provider_name

    return provider


def get_provider(provider_name):
    return _PROVIDERS.get(provider_name)


def list_providers():
    return sorted(_PROVIDERS.keys())


def set_active_provider(provider_name):
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Sportsbook provider is not registered: {provider_name}")

    global _ACTIVE_PROVIDER_NAME
    _ACTIVE_PROVIDER_NAME = provider_name
    return _PROVIDERS[provider_name]


def get_active_provider():
    if _ACTIVE_PROVIDER_NAME is None:
        return None

    return _PROVIDERS.get(_ACTIVE_PROVIDER_NAME)


def clear_providers():
    _PROVIDERS.clear()
    global _ACTIVE_PROVIDER_NAME
    _ACTIVE_PROVIDER_NAME = None


def ensure_default_providers():
    if "mock" not in _PROVIDERS:
        register_provider(MockSportsbookProvider(), make_active=True)

    return _PROVIDERS


def get_provider_health_report():
    ensure_default_providers()

    reports = []
    for provider_name in list_providers():
        provider = get_provider(provider_name)
        try:
            health = provider.health_check()
            projections = provider.fetch_projection_data()
            row_count = int(len(projections))
            coverage = 0.0
            if row_count:
                coverage = round(
                    float(projections["sportsbook_projection"].notna().mean() * 100.0),
                    2,
                )

            reports.append({
                "provider_name": provider_name,
                "status": health.get("status", "unknown"),
                "row_count": row_count,
                "last_updated": (
                    projections.attrs.get("last_updated")
                    or health.get("last_updated")
                ),
                "coverage": coverage,
                "active": get_active_provider() is provider,
            })
        except Exception as exc:
            reports.append({
                "provider_name": provider_name,
                "status": "error",
                "row_count": 0,
                "last_updated": None,
                "coverage": 0.0,
                "active": get_active_provider() is provider,
                "error": str(exc),
            })

    return reports


ensure_default_providers()
