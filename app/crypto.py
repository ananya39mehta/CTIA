import uuid
import hashlib
import base64
from nacl.signing import SigningKey, VerifyKey

# Generate issuer's Ed25519 keypair
issuer_sk = SigningKey.generate()
issuer_vk = issuer_sk.verify_key

def generate_ticket(value: int, upi: str):
    """
    Generate a signed ticket with:
    - Unique ID
    - Hash-lock condition
    - Encrypted UPI (dummy base64 encoding)
    - Ed25519 signature
    """
    ticket_id = str(uuid.uuid4())

    # Create lock condition (hash of secret)
    secret = uuid.uuid4().hex  # random secret
    lock_hash = hashlib.sha256(secret.encode()).hexdigest()

    # Encrypt UPI (dummy: base64 encode)
    vpi_enc = base64.b64encode(upi.encode()).decode()

    # Data to sign
    message = f"{ticket_id}|{value}|{lock_hash}|{vpi_enc}".encode()

    # Sign with Ed25519
    signed = issuer_sk.sign(message)

    ticket = {
        "ticket_id": ticket_id,
        "value": value,
        "lock_hash": lock_hash,
        "vpi_enc": vpi_enc,
        "signature": base64.b64encode(signed.signature).decode(),
    }

    # Include secret (needed for redemption later, but not shared with merchant yet)
    return ticket, secret

def verify_ticket(ticket: dict) -> bool:
    """
    Verify ticket signature using issuer's public key.
    """
    try:
        message = f"{ticket['ticket_id']}|{ticket['value']}|{ticket['lock_hash']}|{ticket['vpi_enc']}".encode()
        signature = base64.b64decode(ticket['signature'])
        issuer_vk.verify(message, signature)
        return True
    except Exception:
        return False
