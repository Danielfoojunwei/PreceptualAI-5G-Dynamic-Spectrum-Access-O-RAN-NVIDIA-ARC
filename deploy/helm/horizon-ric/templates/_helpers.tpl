{{/*
Expand the name of the chart.
*/}}
{{- define "horizon-ric.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully-qualified app name.
*/}}
{{- define "horizon-ric.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Chart label.
*/}}
{{- define "horizon-ric.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "horizon-ric.labels" -}}
helm.sh/chart: {{ include "horizon-ric.chart" . }}
{{ include "horizon-ric.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: o-ran-smo
{{- end -}}

{{/*
Selector labels.
*/}}
{{- define "horizon-ric.selectorLabels" -}}
app.kubernetes.io/name: {{ include "horizon-ric.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Secret name (existing or generated). secret.create=false with no
existingSecret would render a secretRef to a Secret that never exists —
every pod would sit in CreateContainerConfigError — so fail the render
instead of shipping a dangling reference.
*/}}
{{- define "horizon-ric.secretName" -}}
{{- if .Values.secret.existingSecret -}}
{{- .Values.secret.existingSecret -}}
{{- else if not .Values.secret.create -}}
{{- fail "secret.create=false requires secret.existingSecret to be set (the Deployment envFrom references this Secret)" -}}
{{- else -}}
{{- include "horizon-ric.fullname" . -}}-credentials
{{- end -}}
{{- end -}}

{{/*
ConfigMap name.
*/}}
{{- define "horizon-ric.configMapName" -}}
{{- include "horizon-ric.fullname" . -}}-config
{{- end -}}
