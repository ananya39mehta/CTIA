import uuid
import hashlib
import base64
from nacl.signing import SigningKey, VerifyKey

# Generate issuer's Ed25519 keypair (kept in memory for prototype)
issuer_sk = SigningKey.generate()
issuer_vk = issuer_sk.verify_key

def generate_ticket(value: int, upi: str, prob_p_win: float = 0.001):
    """
    Create a ticket dictionary and a secret (preimage).
    Returns (ticket_dict, secret)
    ticket includes prob_p_win.
    """
    ticket_id = str(uuid.uuid4())

    # Create lock condition (hash of secret)
    secret = uuid.uuid4().hex  # random secret (preimage)
    lock_hash = hashlib.sha256(secret.encode()).hexdigest()

    # Encrypt UPI (prototype: base64 encode)
    vpi_enc = base64.b64encode(upi.encode()).decode()

    # Data to sign: include prob_p_win explicitly to fix canonical message
    message = f"{ticket_id}|{value}|{lock_hash}|{vpi_enc}|{prob_p_win}".encode()

    # Sign with Ed25519
    signed = issuer_sk.sign(message)

    ticket = {
        "ticket_id": ticket_id,
        "value": value,
        "lock_hash": lock_hash,
        "vpi_enc": vpi_enc,
        "prob_p_win": prob_p_win,
        "signature": base64.b64encode(signed.signature).decode(),
    }

    return ticket, secret

def verify_ticket(ticket: dict) -> bool:
    """
    Verify ticket signature with issuer's public key.
    """
    try:
        message = f"{ticket['ticket_id']}|{ticket['value']}|{ticket['lock_hash']}|{ticket['vpi_enc']}|{ticket['prob_p_win']}".encode()
        signature = base64.b64decode(ticket['signature'])
        issuer_vk.verify(message, signature)
        return True
    except Exception:
        return False

def is_winner(ticket: dict, preimage: str) -> bool:
    """
    Deterministically check whether a ticket is a winner given the revealed preimage.
    Uses SHA-256(ticket_id || preimage) -> integer -> normalized [0,1). If < prob_p_win => winner.
    """
    # Compute digest over ticket_id || preimage
    h = hashlib.sha256((ticket['ticket_id'] + preimage).encode()).hexdigest()
    # Use first 15 hex digits => up to ~60 bits, convert to integer
    prefix = h[:15]
    val = int(prefix, 16)
    max_val = float(int("f"*15, 16))
    score = val / max_val  # in [0,1)
    return score < float(ticket.get("prob_p_win", 0.001))
