#!/usr/bin/env bash
#
# Cut a small horizontal / temporal box out of a wrfout file, keeping the result
# readable by wrf-python.
#
# Two things make this more than a plain "ncks -d": the staggered horizontal
# dimensions must come out one larger than the unstaggered ones (wrf-python
# infers destaggering from array shape), and the global attributes describing the
# grid size have to be rewritten to match. The vertical dimensions and
# Times/DateStrLen are never touched.
#
# Requires: ncks, ncatted, ncdump (NCO + netCDF).
#
# Usage: subset_wrfout.sh [options] IN.nc OUT.nc
#
#   -s SIZE     horizontal box size, NX or NXxNY   (default 5x5)
#   -c I,J      0-based source index of the box center
#               (default: center of the source domain)
#   -t NT       number of time steps to keep       (default 2)
#   -b T0       0-based first time step            (default 0)
#   -L LEVEL    deflate level, 0 = off             (default 1)
#   -h          this help
#
# Examples:
#   subset_wrfout.sh wrfout_d01_2017-01-02_12:00:00 fixture.nc
#   subset_wrfout.sh -s 9x9 -t 3 wrfout_d01_... fixture.nc
#   subset_wrfout.sh -s 5 -c 40,120 -t 1 -b 5 wrfout_d01_... fixture.nc

set -euo pipefail

SIZE=5x5
CENTER=""
NT=2
T0=0
LEVEL=1

# Print the header comment block above, minus the shebang.
usage() { awk 'NR > 2 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "$0"; }

while getopts ":s:c:t:b:L:h" opt; do
    case "$opt" in
        s) SIZE=$OPTARG ;;
        c) CENTER=$OPTARG ;;
        t) NT=$OPTARG ;;
        b) T0=$OPTARG ;;
        L) LEVEL=$OPTARG ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -ne 2 ]; then
    usage >&2
    exit 2
fi
IN=$1
OUT=$2

for cmd in ncks ncatted ncdump; do
    command -v "$cmd" >/dev/null || { echo "$0: $cmd not found" >&2; exit 1; }
done
[ -r "$IN" ] || { echo "$0: cannot read $IN" >&2; exit 1; }

# --- dimensions of the source file ------------------------------------------

# Size of a fixed dimension, e.g. "west_east = 143 ;". The space before "=" in
# the pattern keeps west_east from also matching west_east_stag.
get_dim() {
    ncdump -h "$IN" | sed -n "s/^[[:space:]]*$1 = \([0-9][0-9]*\) ;.*/\1/p" | head -1
}

# Time is UNLIMITED: "Time = UNLIMITED ; // (6 currently)".
get_time_len() {
    ncdump -h "$IN" | sed -n 's/^[[:space:]]*Time = UNLIMITED ; \/\/ (\([0-9][0-9]*\) currently).*/\1/p' | head -1
}

NX=$(get_dim west_east)
NY=$(get_dim south_north)
NT_SRC=$(get_time_len)
[ -n "$NX" ] && [ -n "$NY" ] && [ -n "$NT_SRC" ] || {
    echo "$0: $IN does not look like a wrfout file (missing west_east/south_north/Time)" >&2
    exit 1
}

# --- requested box ----------------------------------------------------------

case "$SIZE" in
    *x*) SX=${SIZE%x*}; SY=${SIZE#*x} ;;
    *)   SX=$SIZE;      SY=$SIZE ;;
esac

[ "$SX" -ge 1 ] && [ "$SX" -le "$NX" ] && [ "$SY" -ge 1 ] && [ "$SY" -le "$NY" ] || {
    echo "$0: box ${SX}x${SY} does not fit in ${NX}x${NY}" >&2
    exit 1
}

if [ -n "$CENTER" ]; then
    CI=${CENTER%,*}
    CJ=${CENTER#*,}
else
    # Same convention wrf-python uses for the domain center, in 0-based indices.
    # Choosing this keeps CEN_LAT/CEN_LON unchanged for an odd-sized box.
    CI=$(( (NX - 1) / 2 ))
    CJ=$(( (NY - 1) / 2 ))
fi

# Lower corner, with the box shifted inwards if it would hang off an edge.
clamp_lo() {  # center size extent -> lower index
    local lo=$(( $1 - ($2 - 1) / 2 ))
    [ "$lo" -lt 0 ] && lo=0
    [ $(( lo + $2 )) -gt "$3" ] && lo=$(( $3 - $2 ))
    echo "$lo"
}

I0=$(clamp_lo "$CI" "$SX" "$NX")
J0=$(clamp_lo "$CJ" "$SY" "$NY")
I1=$(( I0 + SX - 1 ))
J1=$(( J0 + SY - 1 ))

# Staggered ranges span one extra point: unstaggered i sits between i and i+1.
I1S=$(( I1 + 1 ))
J1S=$(( J1 + 1 ))

# Time range, clipped to what the file actually holds.
T1=$(( T0 + NT - 1 ))
[ "$T1" -ge "$NT_SRC" ] && T1=$(( NT_SRC - 1 ))
[ "$T0" -le "$T1" ] || {
    echo "$0: time step $T0 is past the end of $IN ($NT_SRC steps)" >&2
    exit 1
}

echo "$IN -> $OUT"
echo "  west_east   $I0..$I1  (stag $I0..$I1S)  of $NX"
echo "  south_north $J0..$J1  (stag $J0..$J1S)  of $NY"
echo "  Time        $T0..$T1  of $NT_SRC"
echo "  bottom_top  kept in full"

# --- cut --------------------------------------------------------------------

# Write to a temporary file so an interrupted run cannot leave a truncated
# fixture behind under the real name.
TMP=$(mktemp "${OUT}.XXXXXX") || exit 1
trap 'rm -f "$TMP"' EXIT

ncks -O -4 -L "$LEVEL" \
    -d Time,"$T0","$T1" \
    -d west_east,"$I0","$I1"   -d west_east_stag,"$I0","$I1S" \
    -d south_north,"$J0","$J1" -d south_north_stag,"$J0","$J1S" \
    "$IN" "$TMP"

# --- fix up the grid description -------------------------------------------

# GRID_DIMENSION is the staggered count; PATCH_END_* are 1-based inclusive ends.
ncatted -h -O \
    -a WEST-EAST_GRID_DIMENSION,global,o,i,$((   SX + 1 )) \
    -a SOUTH-NORTH_GRID_DIMENSION,global,o,i,$(( SY + 1 )) \
    -a WEST-EAST_PATCH_END_UNSTAG,global,o,i,"$SX" \
    -a WEST-EAST_PATCH_END_STAG,global,o,i,$((   SX + 1 )) \
    -a SOUTH-NORTH_PATCH_END_UNSTAG,global,o,i,"$SY" \
    -a SOUTH-NORTH_PATCH_END_STAG,global,o,i,$(( SY + 1 )) \
    "$TMP"

# CEN_LAT/CEN_LON are what wrf-python's ll_to_xy/xy_to_ll anchor the grid on, so
# they have to move with the box. Read the center back from XLAT/XLONG of the
# subset: for an even-sized box the center falls between grid points, so average
# the two (or four) straddling ones -- at WRF grid spacings that is sub-meter
# accurate.
#
# TRUELAT1/2, STAND_LON, MOAD_CEN_LAT, POLE_* and DX/DY define the projection
# itself and are deliberately left alone. I_PARENT_START/J_PARENT_START still
# refer to the original domain's position in its parent; they are only nesting
# bookkeeping, not used by wrf-python's coordinate transforms.
center_of() {  # varname -> mean over the straddling center points
    ncks --trd -C -H -s '%.9f\n' -v "$1" \
        -d Time,0,0 \
        -d south_north,$(( (SY - 1) / 2 )),$(( SY / 2 )) \
        -d west_east,$((  (SX - 1) / 2 )),$(( SX / 2 )) \
        "$TMP" \
        | awk 'NF {s += $1; n++} END {if (n) printf "%.9f", s / n}'
}

CEN_LAT=$(center_of XLAT)
CEN_LON=$(center_of XLONG)
if [ -n "$CEN_LAT" ] && [ -n "$CEN_LON" ]; then
    ncatted -h -O \
        -a CEN_LAT,global,o,f,"$CEN_LAT" \
        -a CEN_LON,global,o,f,"$CEN_LON" \
        "$TMP"
    echo "  CEN_LAT/CEN_LON -> $CEN_LAT $CEN_LON"
else
    echo "  warning: no XLAT/XLONG in $IN, CEN_LAT/CEN_LON left unchanged" >&2
fi

mv -f "$TMP" "$OUT"
trap - EXIT
chmod 644 "$OUT"
echo "  wrote $OUT ($(du -h "$OUT" | cut -f1))"
