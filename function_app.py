"""Azure Functions entry point — thin route shell."""

import logging

import azure.functions as func

from asset_master_match.services.match_service import run_asset_master_match

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s:%(lineno)d",
    force=True,
)
logging.getLogger().setLevel(logging.INFO)

app = func.FunctionApp()


@app.route(route="asset_master_match", auth_level=func.AuthLevel.ANONYMOUS)
def asset_master_match(req: func.HttpRequest) -> func.HttpResponse:
    return run_asset_master_match(req)
