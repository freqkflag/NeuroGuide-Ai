import os
import json
from unittest.mock import patch, MagicMock
from backend.agents.brain_dump import BrainDumpAgent

def test_brain_dump_agent_returns_parsed_tasks():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(parsed=[{"title":"Do laundry","priority":"medium"}]))]
    with patch("backend.agents.brain_dump.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = mock_response
        result = BrainDumpAgent.run(input={"text": "I need to do laundry."})
        assert isinstance(result, dict)
        assert "extract_tasks" in result
        assert result["extract_tasks"][0]["title"] == "Do laundry"
