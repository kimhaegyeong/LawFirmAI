<!-- 3b711b23-5369-49fb-b2cb-9dddd6f2a272 895c00ec-693e-4ee8-bf13-1b530b3711d5 -->
# Grafana + Prometheus Monitoring Environment Setup Plan

## Overview

Set up a comprehensive monitoring system for the LawFirmAI project using Grafana and Prometheus with Docker. This will monitor both the law collection scripts (collect_laws.py, collect_laws_optimized.py) and the full system (API, Gradio services) with both real-time and post-analysis capabilities.

## Implementation Steps

### 1. Project Structure Setup

Create new monitoring directory structure:

```
monitoring/
├── docker-compose.yml           # Main monitoring stack
├── prometheus/
│   ├── prometheus.yml          # Prometheus configuration
│   └── rules/
│       └── alerts.yml          # Alert rules
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yml  # Auto-provision Prometheus
│   │   └── dashboards/
│   │       └── dashboard.yml   # Auto-load dashboards
│   └── dashboards/
│       ├── law_collection.json # Collection monitoring
│       └── system_overview.json # System monitoring
└── scripts/
    └── metrics_exporter.py     # Python metrics collector
```

### 2. Docker Configuration

Create `monitoring/docker-compose.yml`:

- Prometheus (port 9090)
- Grafana (port 3000)
- Node Exporter (port 9100) for system metrics
- Persistent volumes for data retention
- Network configuration for service discovery

### 3. Prometheus Setup

Configure `monitoring/prometheus/prometheus.yml`:

- Global scrape interval: 15s
- Scrape configs for:
  - Law collection app (host.docker.internal:8000)
  - FastAPI service (host.docker.internal:8001)
  - Gradio service (host.docker.internal:7860)
  - Node exporter for system metrics
  - Prometheus self-monitoring

Create alert rules in `monitoring/prometheus/rules/alerts.yml`:

- High error rate (>10% over 2 minutes)
- High memory usage (>1GB for 5 minutes)
- Slow processing (>60s 95th percentile for 3 minutes)
- Service unavailable alerts

### 4. Python Metrics Collector

Create `scripts/monitoring/metrics_collector.py`:

- Class: `LawCollectionMetrics`
- Metrics:
  - Counters: pages_processed, laws_collected, errors_total
  - Histograms: page_processing_seconds
  - Gauges: memory_usage, cpu_usage, current_page, throughput
  - Summary: collection_duration
- Background thread for system metrics collection
- HTTP server on port 8000 for Prometheus scraping

### 5. Integration with collect_laws_optimized.py

Modify `scripts/assembly/collect_laws_optimized.py`:

- Import LawCollectionMetrics
- Initialize metrics collector in OptimizedLawCollector.**init**
- Add metrics recording in:
  - save_page(): record page processing time, laws collected, throughput
  - add_failed_item(): record errors by type
  - finalize(): end collection timer
- Add --enable-metrics flag (default: True)
- Store page_processing_time for metrics

### 6. Integration with collect_laws.py

Apply similar modifications to `scripts/assembly/collect_laws.py`:

- Same metrics integration pattern
- Ensure backward compatibility
- Optional metrics flag

### 7. Grafana Dashboard Configuration

Create law collection dashboard (`monitoring/grafana/dashboards/law_collection.json`):

- Stat panels: Pages processed, Laws collected, Current page, Avg processing time
- Time series graphs:
  - Page processing time (50th, 95th, 99th percentiles)
  - Memory usage trend
  - CPU usage trend
  - Throughput (laws per minute)
  - Error rate by type
- Table: Recent errors with details

Create system overview dashboard (`monitoring/grafana/dashboards/system_overview.json`):

- System resource usage (CPU, Memory, Disk)
- Service health status
- Request rates and latencies
- Error rates across services

### 8. Grafana Provisioning

Configure auto-provisioning in `monitoring/grafana/provisioning/`:

- Datasources: Prometheus connection
- Dashboards: Auto-load JSON dashboards
- Default admin credentials: admin/admin123

### 9. Requirements Update

Update `requirements.txt`:

```
prometheus-client>=0.19.0
```

Create separate `monitoring/requirements.txt`:

```
prometheus-client>=0.19.0
psutil>=5.9.0
```

### 10. Documentation

Create `docs/development/monitoring_setup_guide.md`:

- Quick start guide
- Docker compose commands
- Accessing Grafana/Prometheus
- Custom metrics guide
- Alert configuration
- Troubleshooting

Update `docs/development/assembly_data_collection_guide.md`:

- Add monitoring section
- Usage examples with metrics
- Performance analysis workflow

### 11. Startup Scripts

Create `scripts/monitoring/start_monitoring.sh`:

- Start Docker compose stack
- Wait for services to be ready
- Display access URLs
- Health checks

Create `scripts/monitoring/stop_monitoring.sh`:

- Gracefully stop services
- Optional: export data before shutdown

### 12. Post-Analysis Tools

Create `scripts/monitoring/analyze_logs.py`:

- Parse saved metrics from log files
- Generate performance reports
- Compare runs (before/after optimization)
- Export to CSV/JSON for further analysis

### 13. Testing and Validation

- Test metrics collection with sample run (10 pages)
- Verify dashboard updates in real-time
- Test alert triggering
- Validate post-analysis on completed runs

## Key Files to Create/Modify

**New Files:**

- monitoring/docker-compose.yml
- monitoring/prometheus/prometheus.yml
- monitoring/prometheus/rules/alerts.yml
- monitoring/grafana/provisioning/datasources/prometheus.yml
- monitoring/grafana/provisioning/dashboards/dashboard.yml
- monitoring/grafana/dashboards/law_collection.json
- monitoring/grafana/dashboards/system_overview.json
- scripts/monitoring/**init**.py
- scripts/monitoring/metrics_collector.py
- scripts/monitoring/start_monitoring.sh
- scripts/monitoring/stop_monitoring.sh
- scripts/monitoring/analyze_logs.py
- docs/development/monitoring_setup_guide.md
- monitoring/requirements.txt
- monitoring/.env.example

**Modified Files:**

- scripts/assembly/collect_laws_optimized.py (add metrics integration)
- scripts/assembly/collect_laws.py (add metrics integration)
- requirements.txt (add prometheus-client)
- docs/development/assembly_data_collection_guide.md (add monitoring section)

## Success Criteria

- Docker stack starts successfully with all services healthy
- Metrics endpoint accessible at http://localhost:8000/metrics
- Grafana accessible at http://localhost:3000 with pre-loaded dashboards
- Real-time metrics update during law collection
- Alerts trigger correctly when thresholds exceeded
- Post-analysis tools generate comparative reports
- Documentation complete with examples

## Estimated Implementation Time

- Docker and Prometheus setup: 30 minutes
- Metrics collector implementation: 45 minutes
- Script integration: 30 minutes
- Grafana dashboards: 1 hour
- Documentation: 30 minutes
- Testing: 30 minutes

**Total: ~3.5 hours**

### To-dos

- [ ] Create monitoring directory structure with subdirectories for prometheus, grafana, and scripts
- [ ] Create docker-compose.yml with Prometheus, Grafana, and Node Exporter services
- [ ] Configure prometheus.yml with scrape configs and create alert rules
- [ ] Implement metrics_collector.py with Prometheus client for collecting performance metrics
- [ ] Integrate metrics collection into collect_laws_optimized.py
- [ ] Integrate metrics collection into collect_laws.py
- [ ] Create Grafana dashboard JSON files for law collection and system overview
- [ ] Configure Grafana auto-provisioning for datasources and dashboards
- [ ] Create start_monitoring.sh and stop_monitoring.sh scripts
- [ ] Create analyze_logs.py for post-collection performance analysis
- [ ] Update requirements.txt with prometheus-client and create monitoring/requirements.txt
- [ ] Create monitoring_setup_guide.md and update assembly_data_collection_guide.md
- [ ] Test monitoring stack, verify metrics, dashboards, and alerts