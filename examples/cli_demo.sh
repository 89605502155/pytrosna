#!/usr/bin/env sh
# The pytrosna command-line tool. Run: sh examples/cli_demo.sh
set -e
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
cat > "$dir/room.csv" <<CSV
time,temperature,humidity
2026-10-08T10:00:00+03:00,21.5,40
2026-10-08T10:00:10+03:00,21.6,41
2026-10-08T10:00:20+03:00,21.7,41
CSV
pytrosna convert "$dir/room.csv" "$dir/room.trosna" --device room1
pytrosna info "$dir/room.trosna"
pytrosna update "$dir/room.trosna" --time "2026-10-08 10:00:20" temperature=21.65 -m "sensor check"
pytrosna annotate "$dir/room.trosna" --start "2026-10-08 10:00:05" --end "2026-10-08 10:00:15" --label "door open"
pytrosna cat "$dir/room.trosna" --format table
pytrosna cat "$dir/room.trosna" --as-of 1
pytrosna log "$dir/room.trosna"
pytrosna diff "$dir/room.trosna" --from 1
pytrosna verify "$dir/room.trosna"
pytrosna convert "$dir/room.trosna" "$dir/back.csv"
cat "$dir/back.csv"
