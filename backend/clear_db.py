"""Clear all data from the database for a fresh demo recording."""

import asyncio
from sqlalchemy import text
from .database import engine


async def clear_all():
    async with engine.begin() as conn:
        for table in ["experiment_runs", "graph_versions", "graphs", "custom_algorithms"]:
            await conn.execute(text(f"DELETE FROM {table}"))
            label = {
                "experiment_runs": "ExperimentRun",
                "graph_versions": "GraphVersion",
                "graphs": "Graph",
                "custom_algorithms": "CustomAlgorithm",
            }[table]
            print(f"✅ Cleared {label}")
    print("🧹 Database cleared! Ready for demo.")


if __name__ == "__main__":
    asyncio.run(clear_all())
