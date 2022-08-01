#!/usr/bin/env bash
main() {
    screen -S webapp -X quit
    local cnt=0
    while (( $cnt -le 15 ))
    do
        if [ -f updater.lock ]
        then
            sleep 1
            cnt=$(($cnt+1))
        else
            break
        fi
    done
    rm -f updater.lock
}

main
