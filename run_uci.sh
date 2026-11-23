#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export MSYS_NO_PATHCONV=1
export DATASET_NAME=UCI_AirQuality
export REPLAY_HZ="${REPLAY_HZ:-5}"
export REPLAY_LOOP=false

applications=(replay-producer flink-processor synthetic-generator river-uci pipeline5 influx-sink)
consumers=(flink-processor synthetic-generator river-uci pipeline5 influx-sink)
infrastructure=(kafka kafka-ui influxdb grafana)

build_images() {
    docker compose build "${applications[@]}"
}

case "${1:-start}" in
    build)
        build_images
        ;;
    test)
        # Absolute Windows path for MSYS/Git Bash; native path on Linux/macOS.
        root="$(pwd -W 2>/dev/null || pwd)"
        docker compose run --rm --no-deps \
            -v "$root/tests:/app/tests:ro" replay-producer \
            python -m unittest discover -s /app/tests -v
        ;;
    start)
        if [[ "${2:-}" != "" && "${2:-}" != "--build" ]]; then
            echo "Usage: bash run_uci.sh start [--build]" >&2
            exit 2
        fi
        docker compose config --quiet
        if [[ "${2:-}" == "--build" ]]; then build_images; fi
        docker compose stop "${applications[@]}" river-hai
        docker compose up -d --wait "${infrastructure[@]}"
        ready=false
        for ((attempt=0; attempt<30; attempt++)); do
            if docker exec synstream-kafka /opt/kafka/bin/kafka-topics.sh \
                --bootstrap-server kafka:9092 --list >/dev/null 2>&1; then
                ready=true
                break
            fi
            sleep 2
        done
        if [[ "$ready" != true ]]; then
            echo "Kafka did not become ready. Check: docker compose logs kafka" >&2
            exit 1
        fi
        for topic in sensors.raw sensors.windowed.real sensors.features.real \
            sensors.windowed.synthetic analytics.scores analytics.alerts analytics.metrics; do
            docker exec synstream-kafka /opt/kafka/bin/kafka-topics.sh \
                --bootstrap-server kafka:9092 --create --if-not-exists \
                --topic "$topic" --partitions 1 --replication-factor 1
        done
        docker compose up -d --no-deps --force-recreate "${consumers[@]}"
        echo "Allowing consumers 30 seconds to initialize (not a strict readiness check)..."
        sleep 30
        for service in "${consumers[@]}"; do
            id="$(docker compose ps -a -q "$service")"
            if [[ -z "$id" ]] || [[ "$(docker inspect --format '{{.State.Status}}' "$id")" != running ]]; then
                docker compose logs --tail 40 "$service"
                echo "$service is not running; replay was not started." >&2
                exit 1
            fi
        done
        docker compose up -d --no-deps --force-recreate replay-producer
        docker compose ps -a
        echo "UCI started in one-pass mode. Logs: bash run_uci.sh logs"
        echo "Kafka UI: http://localhost:8080 | InfluxDB: http://localhost:8086 | Grafana: http://localhost:3000"
        ;;
    logs)
        docker compose logs -f --tail 20 "${applications[@]}"
        ;;
    status)
        docker compose ps -a
        ;;
    stop)
        docker compose stop "${applications[@]}" "${infrastructure[@]}"
        ;;
    *)
        echo "Usage: bash run_uci.sh {start [--build]|build|test|logs|status|stop}" >&2
        exit 2
        ;;
esac
