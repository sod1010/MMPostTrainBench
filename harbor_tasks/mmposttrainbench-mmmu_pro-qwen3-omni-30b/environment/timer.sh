#!/bin/bash

NUM_HOURS=10
START_FILE="/timer_start"

if [ ! -f "$START_FILE" ]; then
    echo "Timer not initialized (healthcheck has not run yet)."
    exit 1
fi

START_DATE=$(cat "$START_FILE")
DEADLINE=$((START_DATE + NUM_HOURS * 3600))
NOW=$(date +%s)
REMAINING=$((DEADLINE - NOW))

if [ $REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=$((REMAINING / 3600))
    MINUTES=$(((REMAINING % 3600) / 60))
    printf "%d:%02d\n" $HOURS $MINUTES
fi
