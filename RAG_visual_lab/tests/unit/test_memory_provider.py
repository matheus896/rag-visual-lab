import pytest
from unittest.mock import patch
import sys
import os
import fakeredis

# Add the parent directory to the path so we can import from services
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


@pytest.fixture
def fake_redis():
    """
    Pytest fixture that provides a fakeredis client.
    This is a pure-Python implementation of Redis that runs in-memory.
    Each test gets a fresh instance to ensure test isolation.
    """
    return fakeredis.FakeStrictRedis(decode_responses=True)


@patch('redis.Redis')
def test_memory_provider_initialization(mock_redis_class, fake_redis):
    """
    Tests if the MemoryProvider initializes correctly and connects to Redis.
    
    This test patches redis.Redis to return our fakeredis instance,
    ensuring the MemoryProvider uses the in-memory fake Redis instead
    of trying to connect to a real Redis server.
    """
    # Configure the mock to return our fake redis client
    mock_redis_class.return_value = fake_redis
    
    from services.memory_provider import MemoryProvider

    # Initialize the provider
    memory_provider = MemoryProvider(talk_id="test-init")

    # Assert that the Redis client was instantiated with correct parameters
    mock_redis_class.assert_called_once_with(
        host='localhost', 
        port=6379, 
        db=0, 
        decode_responses=True
    )
    
    # Assert the talk_id is set
    assert memory_provider.talk_id == "test-init"

@patch('redis.Redis')
def test_add_message_creates_new_conversation(mock_redis_class, fake_redis):
    """
    Tests if a new conversation is created when add_message is called for a new talk_id.
    
    This test verifies that:
    1. When no conversation exists, a new one is created
    2. The message is correctly formatted with role and content
    3. The conversation is stored in Redis with proper expiration
    """
    # Configure the mock to return our fake redis client
    mock_redis_class.return_value = fake_redis
    
    from services.memory_provider import MemoryProvider

    memory_provider = MemoryProvider(talk_id="new-convo")
    memory_provider.add_message("user", "Hello, world!")

    # Verify that the conversation was stored in Redis
    stored_data = fake_redis.get("conversation:new-convo")
    assert stored_data is not None
    
    # Verify the JSON structure
    import json
    conversation = json.loads(stored_data)
    assert len(conversation) == 1
    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"] == "Hello, world!"

@patch('redis.Redis')
def test_add_message_updates_existing_conversation(mock_redis_class, fake_redis):
    """
    Tests if an existing conversation is correctly updated.
    
    This test verifies that:
    1. When a conversation exists, new messages are appended to it
    2. The order of messages is preserved (newest first)
    3. All existing messages are retained
    """
    # Configure the mock to return our fake redis client
    mock_redis_class.return_value = fake_redis
    
    import json
    from services.memory_provider import MemoryProvider
    
    # Pre-populate Redis with existing conversation
    existing_history = [{"role": "user", "content": "First message"}]
    fake_redis.set("conversation:existing-convo", json.dumps(existing_history))

    memory_provider = MemoryProvider(talk_id="existing-convo")
    memory_provider.add_message("assistant", "Second message")

    # Verify the conversation was updated
    stored_data = fake_redis.get("conversation:existing-convo")
    updated_history = json.loads(stored_data)
    
    assert len(updated_history) == 2
    # New message should be first (prepended)
    assert updated_history[0]["role"] == "assistant"
    assert updated_history[0]["content"] == "Second message"
    # Old message should be second
    assert updated_history[1]["role"] == "user"
    assert updated_history[1]["content"] == "First message"

@patch('redis.Redis')
def test_get_conversation_returns_history(mock_redis_class, fake_redis):
    """
    Tests that get_conversation correctly retrieves and deserializes the history.
    
    This test verifies that:
    1. Stored conversation data is correctly retrieved from Redis
    2. JSON deserialization works properly
    3. The returned structure matches the expected format
    """
    # Configure the mock to return our fake redis client
    mock_redis_class.return_value = fake_redis
    
    import json
    from services.memory_provider import MemoryProvider
    
    # Pre-populate Redis with conversation data
    history_data = [{"role": "user", "content": "Test message"}]
    fake_redis.set("conversation:get-test", json.dumps(history_data))

    memory_provider = MemoryProvider(talk_id="get-test")
    conversation = memory_provider.get_conversation()

    assert conversation == history_data
    assert len(conversation) == 1
    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"] == "Test message"

@patch('redis.Redis')
def test_get_conversation_returns_none_for_invalid_id(mock_redis_class, fake_redis):
    """
    Tests that get_conversation returns None when no history exists.
    
    This test verifies that:
    1. When no conversation is stored in Redis, None is returned
    2. The method handles missing keys gracefully
    3. No errors are raised for non-existent conversations
    """
    # Configure the mock to return our fake redis client
    mock_redis_class.return_value = fake_redis
    
    from services.memory_provider import MemoryProvider

    # Don't populate Redis - the conversation doesn't exist
    memory_provider = MemoryProvider(talk_id="invalid-id")
    conversation = memory_provider.get_conversation()

    assert conversation is None

@patch('redis.Redis')
def test_delete_conversation_removes_history(mock_redis_class, fake_redis):
    """
    Tests that the delete_conversation method correctly removes history from Redis.
    
    This test verifies that:
    1. The delete method properly removes the conversation key from Redis
    2. After deletion, the conversation cannot be retrieved
    3. The deletion operation completes without errors
    """
    # Configure the mock to return our fake redis client
    mock_redis_class.return_value = fake_redis
    
    import json
    from services.memory_provider import MemoryProvider
    
    # Pre-populate Redis with conversation data
    history_data = [{"role": "user", "content": "To be deleted"}]
    fake_redis.set("conversation:delete-test", json.dumps(history_data))
    
    # Verify the data exists before deletion
    assert fake_redis.get("conversation:delete-test") is not None

    memory_provider = MemoryProvider(talk_id="delete-test")
    memory_provider.delete_conversation()

    # Verify the conversation was deleted
    assert fake_redis.get("conversation:delete-test") is None
