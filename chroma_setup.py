import chromadb
from chromadb.config import Settings
from chromadb.api.client import Client  # Import the Client class
from datetime import datetime, timedelta


def initialize_client():
    client = chromadb.PersistentClient(
        path="app/chromadb",  # Specify the relative path to the database directory
        settings=Settings(),
        tenant="default_tenant",  # Use your intended tenant
        database="default_database",  # Set a default database if needed
    )
    return client


def create_tenant(client, tenant_name):
    try:
        client.create_tenant(name=tenant_name)
        print(f"Tenant {tenant_name} created successfully.")
    except Exception as e:
        print(f"Error creating tenant {tenant_name}: {e}")


def ensure_tenant(client, tenant_name):
    try:
        if not client.get_tenant(tenant_name):
            create_tenant(client, tenant_name)
    except Exception as e:
        print(f"Error checking tenant {tenant_name}: {e}")
