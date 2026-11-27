"""Build-time check of the ordinary entry and SK_BIND in a linked WeLM ELF.

Uses only the standard library; it can also inspect a collected device .o on a
machine without CANN. ELF64 section/symbol records and CANN's SK_INFO TLV are
read directly, so msobjdump/disassembly support is not required.
"""

import argparse
import struct
from pathlib import Path


def check_elf(path, kernel):
    data = Path(path).read_bytes()
    if data[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<HH", data, 16) != (2, 0x1029):
        raise ValueError("expected a linked little-endian Ascend ELF64 device binary")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, name_index = struct.unpack_from("<HHH", data, 58)
    if size != 64:
        raise ValueError("unexpected ELF section record size")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)]

    def contents(section):
        return data[section[4]:section[4] + section[5]]

    def string(table, start):
        return table[start:table.index(b"\0", start)].decode()

    names = contents(sections[name_index])
    named = {string(names, s[0]): s for s in sections}
    symbols = {}
    table = named[".symtab"]
    strings = contents(sections[table[6]])
    for pos in range(table[4], table[4] + table[5], table[9]):
        name, info, _, section, value, length = struct.unpack_from("<IBBHQQ", data, pos)
        if info & 15 == 2 and section:  # Defined STT_FUNC, not a declaration.
            symbols[string(strings, name)] = (value, info >> 4, length)
    if kernel not in symbols or symbols[kernel][1] != 1 or symbols[kernel][2] == 0:
        raise ValueError(f"missing ordinary STB_GLOBAL entry {kernel}; auto_gen renaming is not allowed")
    if any(n.startswith(kernel + "_origin") for n in symbols):
        raise ValueError("legacy *_origin wrapper is still present")

    def tlvs(section):
        payload = contents(named[section])
        pos = 0
        while pos < len(payload):
            kind, length = struct.unpack_from("<HH", payload, pos)
            end = pos + 4 + length
            if end > len(payload):
                raise ValueError(f"truncated {section} TLV")
            yield kind, payload[pos + 4:end]
            pos = end

    metadata = list(tlvs(".ascend.meta." + kernel))
    if (1, struct.pack("<I", 3)) not in metadata:  # F_TYPE_KTYPE = K_TYPE_AIV
        raise ValueError("ordinary entry lacks AIV kernel-type metadata")
    bindings = [p for kind, p in tlvs(".ascend.meta") if kind == 5]  # B_TYPE_SK_INFO
    if len(bindings) != 1 or len(bindings[0]) != 52:
        raise ValueError("expected one SK_BIND with four SK functions")
    # Reserved uint32, capability uint64, global pointer, four SK pointers.
    capability, global_address, *sk_addresses = struct.unpack_from("<6Q", bindings[0], 4)
    if capability != 0 or global_address != symbols[kernel][0]:
        raise ValueError(f"SK_BIND targets {global_address:#x}, not ordinary entry {symbols[kernel][0]:#x}")
    sk_name = kernel.removesuffix("_kernel") + "_sk"
    for split, address in enumerate(sk_addresses):
        matches = [v for n, v in symbols.items() if sk_name in n and f"ILj{split}E" in n]
        if len(matches) != 1 or matches[0][0] != address or address == global_address:
            raise ValueError(f"SK_BIND specialization {split} does not identify its distinct SK entry")
    return f"{kernel}: ordinary={global_address:#x}, SK={[hex(a) for a in sk_addresses]}, binding OK"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--kernel", required=True)
    args = parser.parse_args()
    try:
        print(check_elf(args.binary, args.kernel))
    except (ValueError, KeyError, IndexError, struct.error) as error:
        raise SystemExit(f"Invalid WeLM device ELF {args.binary}: {error}") from error
