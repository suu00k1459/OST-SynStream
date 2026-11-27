#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export MSYS_NO_PATHCONV=1
dataset="${1:?Expected uci or hai}"
shift
case "$dataset" in
    uci)
        export DATASET_NAME=UCI_AirQuality
        export MODEL_FILE=timevae_uci.pt
        export REPLAY_HZ="${REPLAY_HZ:-5}"
        river=river-uci
        extras=(feature-summaries)
        ;;
    hai)
        export DATASET_NAME=HAI
        export MODEL_FILE=timevae_hai.pt
        export REPLAY_HZ="${REPLAY_HZ:-100}"
        river=river-hai
        extras=(feature-summaries alert-eval)
        ;;
    *) echo "Expected uci or hai" >&2; exit 2 ;;
esac
export REPLAY_LOOP=false
launcher="run_${dataset}.sh"

applications=(replay-producer flink-processor synthetic-generator "$river" pipeline5 influx-sink "${extras[@]}")
consumers=(flink-processor synthetic-generator "$river" pipeline5 influx-sink "${extras[@]}")
all_applications=(replay-producer flink-processor synthetic-generator river-uci river-hai pipeline5 influx-sink feature-summaries alert-eval)
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
            -v "$root:/workspace:ro" -w /workspace \
            -e DATASET_NAME=UCI_AirQuality replay-producer \
            python -m unittest discover -s tests -v
        ;;
    start)
        if [[ "${2:-}" != "" && "${2:-}" != "--build" ]]; then
            echo "Usage: bash $launcher start [--build]" >&2
            exit 2
        fi
        if [[ ! -f "synthetic_gen/models/$MODEL_FILE" ]]; then
            echo "Missing checkpoint: synthetic_gen/models/$MODEL_FILE" >&2
            exit 1
        fi
        if [[ "$dataset" == hai && ! -f data/hai-22.04/test1.csv ]]; then
            echo "Missing local HAI file: data/hai-22.04/test1.csv" >&2
            exit 1
        fi
        docker compose config --quiet
        if [[ "${2:-}" == "--build" ]]; then build_images; fi
        docker compose stop "${all_applications[@]}"
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
        echo "$DATASET_NAME started in one-pass mode. Logs: bash $launcher logs"
        echo "Kafka UI: http://localhost:8080 | InfluxDB: http://localhost:8086 | Grafana: http://localhost:3000"
        ;;
    logs)
        docker compose logs -f --tail 20 "${applications[@]}"
        ;;
    status)
        docker compose ps -a
        ;;
    stop)
        docker compose stop "${all_applications[@]}" "${infrastructure[@]}"
        ;;
    *)
        echo "Usage: bash $launcher {start [--build]|build|test|logs|status|stop}" >&2
        exit 2
        ;;
esac
