"""BSON helper utilities."""
import json
from bson import ObjectId
from datetime import datetime, date
import bson

class BSONEncoder(json.JSONEncoder):
    """JSON encoder that handles BSON types like ObjectId and datetime."""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, bson.Binary):
            return str(obj)
        if isinstance(obj, bson.Decimal128):
            return float(obj)
        if isinstance(obj, bson.Int64):
            return int(obj)
        if isinstance(obj, bson.MaxKey):
            return "MaxKey"
        if isinstance(obj, bson.MinKey):
            return "MinKey"
        if isinstance(obj, bson.Timestamp):
            return {"t": obj.time, "i": obj.inc}
        return super().default(obj)

def parse_bson_to_json(bson_data):
    """Convert BSON data to JSON-serializable format."""
    return json.loads(json.dumps(bson_data, cls=BSONEncoder))