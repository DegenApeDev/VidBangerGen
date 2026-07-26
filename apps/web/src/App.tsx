/// <reference types="vite/client" />

import { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';

const API = import.meta.env.DEV ? '/api' : '';

type Json = Record<string, any>;
type Workspace = 'studio' | 'quick';
type PromptMode = 'assisted' | 'manual';
type AudioMode = 'prompt' | 'ambient' | 'silent' | 'native-dialogue';
type StudioAudioPolicy = 'shot' | 'ambient' | 'silent' | 'native-dialogue';
type TransformMode = 'day-to-night' | 'deblur' | 'decompression' | 'colorization' | 'clean-plate' | 'foley-v2a' | 'water-simulation' | 'instant-shave' | 'cross-eyed';
type ExecutionPreferences = { generation: string; continuation: string; postUpscale: string };

const ACTIVE_JOBS = new Set(['queued', 'running']);
const QUICK_ACTIVE_KEY = 'vbg-quick-active';
const QUICK_CHAIN_KEY = 'vbg-quick-chain';
const AUDIO_POLICY_REVISION_KEY = 'vbg-audio-policy-revision';
const AUDIO_POLICY_REVISION = 'ambient-default-v2';
const NEW_PROJECT_DRAFT_KEY = 'vbg-new-project-draft';
const EXECUTION_PREFS_KEY = 'vbg-execution-preferences';
const EXECUTION_PREFS_REVISION_KEY = 'vbg-execution-preferences-revision';
const EXECUTION_PREFS_REVISION = 'pixel-spatial-primary-v1';
const RESOLUTIONS = [
  { label: 'Vertical', detail: '720 × 1280 output', width: 360, height: 640 },
  { label: 'Landscape', detail: '1280 × 720 output', width: 640, height: 360 },
  { label: 'Square', detail: '1024 × 1024 output', width: 512, height: 512 },
  { label: 'Motion draft', detail: '512 × 896 output', width: 256, height: 448 },
];
const TRANSFORM_UI: Record<TransformMode, { label: string; action: string; placeholder: string; strength?: number; promptRequired?: boolean }> = {
  'water-simulation': { label: 'Water Simulation', action: 'Add water', placeholder: 'Required: water type, motion, surfaces, and interaction…', strength: 1.2, promptRequired: true },
  'instant-shave': { label: 'Instant Shave', action: 'Remove facial hair', placeholder: 'Optional: describe the clean-shaven subject and scene…' },
  'cross-eyed': { label: 'Cross-Eyed', action: 'Apply eye effect', placeholder: 'Optional: describe the close-up portrait subject…' },
  'day-to-night': { label: 'Day → Night', action: 'Relight', placeholder: 'Optional night lighting direction…' },
  deblur: { label: 'Deblur', action: 'Restore focus', placeholder: 'Describe the source scene for identity lock…' },
  decompression: { label: 'Decompress', action: 'Remove artifacts', placeholder: 'Describe the source scene and important fine detail…' },
  colorization: { label: 'Colorize', action: 'Restore color', placeholder: 'Describe the intended natural colors and materials…' },
  'clean-plate': { label: 'Clean Plate', action: 'Remove subjects', placeholder: 'Describe the empty location and name anything that must disappear…' },
  'foley-v2a': { label: 'Foley V2A', action: 'Generate sound', placeholder: 'Optional: footsteps, impacts, surfaces, materials, ambience…' },
};

function savedExecutionPreferences(): ExecutionPreferences {
  try {
    const value = { generation: 'auto', continuation: 'auto', postUpscale: 'primary', ...JSON.parse(localStorage.getItem(EXECUTION_PREFS_KEY) || '{}') };
    if (localStorage.getItem(EXECUTION_PREFS_REVISION_KEY) !== EXECUTION_PREFS_REVISION) {
      value.postUpscale = 'primary';
      localStorage.setItem(EXECUTION_PREFS_REVISION_KEY, EXECUTION_PREFS_REVISION);
      localStorage.setItem(EXECUTION_PREFS_KEY, JSON.stringify(value));
    }
    return value;
  } catch {
    return { generation: 'auto', continuation: 'auto', postUpscale: 'primary' };
  }
}

function executionLabel(info: Json | null, id: string, fallback = 'Automatic routing') {
  if (id === 'auto') return fallback;
  return info?.targets?.find((target: Json) => target.id === id)?.label || id;
}

function savedQuickGeneration() {
  try {
    const value = JSON.parse(localStorage.getItem(QUICK_ACTIVE_KEY) || 'null');
    if (!value?.promptId || Date.now() - Number(value.startedAt || 0) > 45 * 60 * 1000) {
      localStorage.removeItem(QUICK_ACTIVE_KEY);
      return null;
    }
    return value;
  } catch {
    localStorage.removeItem(QUICK_ACTIVE_KEY);
    return null;
  }
}

function savedQuickChain() {
  try {
    const value = JSON.parse(localStorage.getItem(QUICK_CHAIN_KEY) || 'null');
    if (!value?.chainId || Date.now() - Number(value.savedAt || 0) > 30 * 24 * 60 * 60 * 1000) {
      localStorage.removeItem(QUICK_CHAIN_KEY);
      return null;
    }
    return value;
  } catch {
    localStorage.removeItem(QUICK_CHAIN_KEY);
    return null;
  }
}

function migratedQuickAudioMode(value?: AudioMode): AudioMode {
  try {
    if (localStorage.getItem(AUDIO_POLICY_REVISION_KEY) !== AUDIO_POLICY_REVISION) {
      localStorage.setItem(AUDIO_POLICY_REVISION_KEY, AUDIO_POLICY_REVISION);
      // Move old silent-default browser state to the new sound-on default while
      // preserving unmistakably deliberate dialogue/advanced selections.
      return value === 'native-dialogue' || value === 'prompt' ? value : 'ambient';
    }
  } catch {
    // Storage can be unavailable in privacy modes; the sound-on default still wins.
  }
  return value || 'ambient';
}

function savedProjectDraft() {
  try {
    return JSON.parse(localStorage.getItem(NEW_PROJECT_DRAFT_KEY) || 'null');
  } catch {
    localStorage.removeItem(NEW_PROJECT_DRAFT_KEY);
    return null;
  }
}

async function request(path: string, init?: RequestInit) {
  const response = await fetch(`${API}${path}`, init);
  const contentType = response.headers.get('content-type') || '';
  const value = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof value === 'object' ? value.detail || value.message : value;
    const failure = new Error(
      typeof detail === 'string' ? detail : JSON.stringify(detail || `Request failed (${response.status})`)
    ) as Error & { status: number };
    failure.status = response.status;
    throw failure;
  }
  return value;
}

function relativeTime(value?: string) {
  if (!value) return '';
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1000));
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function prettyStatus(value = 'draft') {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter: string) => letter.toUpperCase());
}

function artifactName(file: any) {
  if (!file) return '';
  if (typeof file === 'string') return file;
  return [file.subfolder, file.filename].filter(Boolean).join('/');
}

function quotedWords(value: string) {
  return Array.from(value.matchAll(/[“"]([^”"]+)[”"]/g))
    .map((match) => match[1].trim()).filter(Boolean).join(' ');
}

function outputUrl(filename: string) {
  return `${API}/output/${filename.split('/').map(encodeURIComponent).join('/')}`;
}

function ChainClipVideo({ chainId, clip }: { chainId: string; clip: Json }) {
  const durableSource = `${API}/chain/${encodeURIComponent(chainId)}/clips/${encodeURIComponent(clip.id)}/output`;
  const remoteSource = typeof clip.remote_filename === 'string' && clip.remote_filename
    ? outputUrl(clip.remote_filename)
    : '';
  const [source, setSource] = useState(durableSource);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setSource(durableSource);
    setFailed(false);
  }, [durableSource, remoteSource]);

  function handleError() {
    // Vite can hot-reload this UI while a non-reloading API process is still
    // serving the previous route set. The established remote output route is
    // a safe preview fallback until that API process is restarted.
    if (remoteSource && source !== remoteSource) {
      setSource(remoteSource);
      return;
    }
    setFailed(true);
  }

  if (failed) {
    return <div className="chain-preview-error"><span>Preview could not load</span><button type="button" onClick={() => { setFailed(false); setSource(durableSource); }}>Retry preview</button></div>;
  }

  return <video key={source} controls preload="metadata" src={source} onError={handleError} />;
}

function Pill({ status }: { status?: string }) {
  const value = status || 'draft';
  return <span className={`pill pill-${value}`}>{prettyStatus(value)}</span>;
}

function EmptyState({ title, children }: { title: string; children: React.ReactNode }) {
  return <div className="empty-state"><div className="empty-mark">V</div><h3>{title}</h3><p>{children}</p></div>;
}

function App() {
  const [workspace, setWorkspace] = useState<Workspace>('studio');
  const [projects, setProjects] = useState<Json[]>([]);
  const [projectId, setProjectId] = useState(() => localStorage.getItem('vbg-project') || '');
  const [project, setProject] = useState<Json | null>(null);
  const [health, setHealth] = useState<Json | null>(null);
  const [models, setModels] = useState<Json | null>(null);
  const [creativeLab, setCreativeLab] = useState<Json | null>(null);
  const [executionInfo, setExecutionInfo] = useState<Json | null>(null);
  const [execution, setExecution] = useState<ExecutionPreferences>(() => savedExecutionPreferences());
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [showGpuMonitor, setShowGpuMonitor] = useState(false);
  const [showExecution, setShowExecution] = useState(false);

  function saveExecution(value: ExecutionPreferences) {
    setExecution(value);
    localStorage.setItem(EXECUTION_PREFS_KEY, JSON.stringify(value));
  }

  const loadProjects = useCallback(async () => {
    const value = await request('/projects');
    setProjects(value.projects || []);
    return value.projects || [];
  }, []);

  const loadProject = useCallback(async (id = projectId) => {
    if (!id) { setProject(null); return; }
    const value = await request(`/projects/${id}`);
    setProject(value);
  }, [projectId]);

  const boot = useCallback(async () => {
    try {
      const [list, modelInfo, labInfo, targetInfo] = await Promise.all([loadProjects(), request('/models'), request('/creative-lab').catch(() => null), request('/execution-targets').catch(() => null)]);
      setModels(modelInfo);
      setCreativeLab(labInfo);
      setExecutionInfo(targetInfo);
      const remembered = projectId && list.some((item: Json) => item.id === projectId);
      if (!remembered && list.length) setProjectId(list[0].id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to load VidBangerGen');
    }
    request('/health').then(setHealth).catch(() => setHealth({ status: 'degraded' }));
  }, [loadProjects, projectId]);

  useEffect(() => { boot(); }, []);
  useEffect(() => {
    if (!executionInfo?.targets?.length) return;
    const has = (id: string, capability: string) => id === 'auto' || executionInfo.targets.some((target: Json) => target.id === id && target.capabilities?.includes(capability));
    const hasAvailable = (id: string, capability: string) => executionInfo.targets.some((target: Json) => target.id === id && target.available && target.capabilities?.includes(capability));
    const postTargets = executionInfo.targets.filter((target: Json) => target.capabilities?.includes('post-upscale'));
    const normalized: ExecutionPreferences = {
      generation: has(execution.generation, 'ltx-generation') ? execution.generation : 'auto',
      continuation: has(execution.continuation, 'continuation') ? execution.continuation : 'auto',
      postUpscale: hasAvailable(execution.postUpscale, 'post-upscale') ? execution.postUpscale : (executionInfo.defaults?.post_upscale || postTargets.find((target: Json) => target.available)?.id || postTargets[0]?.id || ''),
    };
    if (JSON.stringify(normalized) !== JSON.stringify(execution)) saveExecution(normalized);
  }, [executionInfo]);
  useEffect(() => {
    if (!projectId) { setProject(null); return; }
    localStorage.setItem('vbg-project', projectId);
    loadProject(projectId).catch((cause) => setError(cause.message));
  }, [projectId, loadProject]);

  const hasActiveJobs = Boolean(project?.jobs?.some((job: Json) => ACTIVE_JOBS.has(job.status)));
  useEffect(() => {
    if (!projectId) return;
    const interval = window.setInterval(() => {
      loadProject(projectId).catch(() => undefined);
      if (hasActiveJobs) loadProjects().catch(() => undefined);
    }, hasActiveJobs ? 2500 : 12000);
    return () => window.clearInterval(interval);
  }, [projectId, hasActiveJobs, loadProject, loadProjects]);

  async function act(label: string, operation: () => Promise<any>, success?: string) {
    setBusy(label); setError(''); setNotice('');
    try {
      const value = await operation();
      if (projectId) await loadProject(projectId);
      await loadProjects();
      if (success) setNotice(success);
      return value;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Something went wrong');
      throw cause;
    } finally {
      setBusy('');
    }
  }

  const workerHealthy = health?.workers?.filter((worker: Json) => worker.healthy).length || 0;
  const workerTotal = health?.workers?.length || models?.workers?.length || 0;

  return (
    <div className="shell">
      <header className="topbar">
        <button className="brand" onClick={() => setWorkspace('studio')}>
          <span className="brand-mark">VB</span>
          <span><strong>VidBangerGen</strong><small>LTX 2.3 Production Studio</small></span>
        </button>
        <nav className="workspace-tabs" aria-label="Workspace">
          <button className={workspace === 'studio' ? 'active' : ''} onClick={() => setWorkspace('studio')}>Studio</button>
          <button className={workspace === 'quick' ? 'active' : ''} onClick={() => setWorkspace('quick')}>Quick Generate</button>
        </nav>
        <div className="topbar-status">
          <button className="compute-target-button" onClick={() => setShowExecution(true)}><span>Compute</span>{executionLabel(executionInfo, execution.generation)}</button>
          <button className="gpu-monitor-button" onClick={() => setShowGpuMonitor(true)}><span>GPU</span> Monitor</button>
          <div className={`worker-state ${workerHealthy ? 'online' : 'offline'}`}>
            <i /> {workerHealthy ? `${workerHealthy}/${workerTotal} workers ready` : 'Inference unavailable'}
          </div>
        </div>
      </header>

      {error && <div className="toast toast-error"><span>{error}</span><button onClick={() => setError('')}>×</button></div>}
      {notice && <div className="toast toast-success"><span>{notice}</span><button onClick={() => setNotice('')}>×</button></div>}

      <div className={workspace === 'quick' ? 'workspace-page' : 'workspace-page workspace-hidden'}>
        <QuickGenerate onError={setError} execution={execution} executionInfo={executionInfo} />
      </div>

      <div className={workspace === 'studio' ? 'studio-layout' : 'studio-layout workspace-hidden'}>
          <aside className="project-rail">
            <div className="rail-head"><span>Projects</span><button className="icon-button" onClick={() => setCreating(true)} aria-label="New project">+</button></div>
            <div className="project-list">
              {projects.map((item) => (
                <button key={item.id} className={`project-item ${item.id === projectId ? 'active' : ''}`} onClick={() => setProjectId(item.id)}>
                  <span className="project-thumb">{(item.title || 'V').slice(0, 1).toUpperCase()}</span>
                  <span className="project-copy"><strong>{item.title}</strong><small>{prettyStatus(item.status)} · {relativeTime(item.updated_at)}</small></span>
                </button>
              ))}
              {!projects.length && <p className="rail-empty">No projects yet.</p>}
            </div>
            <button className="new-project-button" onClick={() => setCreating(true)}>+ New project</button>
          </aside>

          <main className="studio-main">
            {!project ? (
              <EmptyState title="Build your first banger">Create a project, choose human-authored prompts or the Bonsai creative director, and move from storyboard to finished export.</EmptyState>
            ) : (
              <ProjectStudio
                key={project.id}
                project={project}
                models={models}
                creativeLab={creativeLab}
                execution={execution}
                executionInfo={executionInfo}
                onExecutionChange={saveExecution}
                busy={busy}
                refresh={() => loadProject(project.id)}
                act={act}
              />
            )}
          </main>
      </div>

      {creating && <NewProject backendFeatures={models?.features || []} onClose={() => setCreating(false)} onCreated={async (value) => {
        setCreating(false); await loadProjects(); setProjectId(value.id); setNotice('Project created');
      }} />}
      {showGpuMonitor && <GpuMonitor onClose={() => setShowGpuMonitor(false)} />}
      {showExecution && <ExecutionModal info={executionInfo} value={execution} onChange={saveExecution} onClose={() => setShowExecution(false)} onRefresh={() => request('/execution-targets').then(setExecutionInfo).catch((cause) => setError(cause.message))} />}
    </div>
  );
}

function GpuMonitor({ onClose }: { onClose: () => void }) {
  const [status, setStatus] = useState<Json | null>(null);
  const [error, setError] = useState('');
  const [refreshing, setRefreshing] = useState(true);

  const load = useCallback(async () => {
    try {
      const value = await request('/gpu-status');
      setStatus(value); setError('');
    } catch (cause) {
      const failure = cause as Error & { status?: number };
      setError(failure.status === 404
        ? 'Remote GPU monitoring is not configured. Run the setup wizard again and provide an SSH target.'
        : failure.message);
    } finally { setRefreshing(false); }
  }, []);

  useEffect(() => {
    load();
    const interval = window.setInterval(load, 3000);
    return () => window.clearInterval(interval);
  }, [load]);

  return <div className="modal-backdrop gpu-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section className="modal gpu-modal" role="dialog" aria-modal="true" aria-label="Remote GPU monitor">
      <div className="modal-head"><div><span className="eyebrow">Live remote telemetry</span><h2>GPU Monitor</h2><p>Read-only NVIDIA metrics · refreshes every 3 seconds while open</p></div><button type="button" className="icon-button" onClick={onClose}>×</button></div>
      {error && <div className="gpu-monitor-error"><strong>Telemetry unavailable</strong><span>{error}</span></div>}
      {!status && refreshing && <div className="gpu-monitor-loading"><i /><span>Reading nvidia-smi over SSH…</span></div>}
      {status?.gpus?.length ? <div className="gpu-card-grid">{status.gpus.map((gpu: Json) => {
        const memory = Number(gpu.memory_percent || 0);
        const utilization = Number(gpu.utilization_percent || 0);
        const temperature = Number(gpu.temperature_c || 0);
        const power = gpu.power_limit_w ? Math.min(100, Number(gpu.power_draw_w || 0) / Number(gpu.power_limit_w) * 100) : 0;
        const activity = utilization >= 10 ? 'Rendering' : memory >= 45 ? 'Model loaded · between passes' : memory >= 8 ? 'Preparing / decoding' : 'Idle';
        return <article className="gpu-card" key={gpu.index}>
          <div className="gpu-card-head"><div><span>GPU {gpu.index}</span><h3>{gpu.name}</h3></div><strong className={utilization >= 10 ? 'active' : memory >= 45 ? 'resident' : ''}><i />{activity}</strong></div>
          <div className="gpu-primary-metrics"><div><span>VRAM</span><strong>{(Number(gpu.memory_used_mib || 0) / 1024).toFixed(1)}<small> / {(Number(gpu.memory_total_mib || 0) / 1024).toFixed(0)} GB</small></strong></div><div className={temperature >= 80 ? 'metric-hot' : temperature >= 70 ? 'metric-warm' : ''}><span>Temperature</span><strong>{gpu.temperature_c ?? '—'}<small>°C</small></strong></div><div><span>GPU load</span><strong>{gpu.utilization_percent ?? '—'}<small>%</small></strong></div></div>
          <GpuMeter label="VRAM allocation" value={memory} tone="violet" />
          <GpuMeter label="Compute utilization" value={utilization} tone="lime" />
          <GpuMeter label="Power draw" value={power} tone="cyan" detail={gpu.power_draw_w == null ? 'N/A' : `${Math.round(gpu.power_draw_w)} / ${Math.round(gpu.power_limit_w)} W`} />
          <div className="gpu-secondary"><span>Fan <b>{gpu.fan_percent ?? 'N/A'}{gpu.fan_percent == null ? '' : '%'}</b></span><span>Performance <b>{gpu.performance_state || 'N/A'}</b></span><span>Driver <b>{gpu.driver_version || 'N/A'}</b></span></div>
        </article>;
      })}</div> : null}
      <div className="gpu-monitor-footer"><p>VRAM can stay high while utilization falls to 0% between LTX stages. That usually means the model remains loaded, not that the render failed.</p><div><span>{status?.updated_at ? `Updated ${new Date(status.updated_at).toLocaleTimeString()}` : 'Waiting for telemetry'}</span><button className="button secondary" onClick={() => { setRefreshing(true); load(); }} disabled={refreshing}>{refreshing ? 'Refreshing…' : 'Refresh now'}</button></div></div>
    </section>
  </div>;
}

function ExecutionModal({ info, value, onChange, onClose, onRefresh }: { info: Json | null; value: ExecutionPreferences; onChange: (value: ExecutionPreferences) => void; onClose: () => void; onRefresh: () => void }) {
  const targets = info?.targets || [];
  const generationTargets = targets.filter((target: Json) => target.capabilities?.includes('ltx-generation'));
  const continuationTargets = targets.filter((target: Json) => target.capabilities?.includes('continuation'));
  const upscaleTargets = targets.filter((target: Json) => target.capabilities?.includes('post-upscale'));
  function select(key: keyof ExecutionPreferences, selected: string) {
    onChange({ ...value, [key]: selected });
  }
  return <div className="modal-backdrop execution-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section className="modal execution-modal" role="dialog" aria-modal="true" aria-label="Execution routing">
      <div className="modal-head"><div><span className="eyebrow">Portable compute routing</span><h2>Choose where each stage runs</h2><p>Routes are saved in this browser and attached to every new job. Existing jobs stay on their original machine.</p></div><button type="button" className="icon-button" onClick={onClose}>×</button></div>
      <div className="execution-routes">
        <label><span>LTX generation + latent upscale</span><select value={value.generation} onChange={(event) => select('generation', event.target.value)}><option value="auto">Auto · first healthy compatible pool</option>{generationTargets.map((target: Json) => <option key={target.id} value={target.id} disabled={!target.available}>{target.label}{target.available ? '' : ' · offline'}</option>)}</select><small>The current 4-step motion and 3-step latent upscale stay together on this LTX worker.</small></label>
        <label><span>Story continuations</span><select value={value.continuation} onChange={(event) => select('continuation', event.target.value)}><option value="auto">Auto · upload-capable worker</option>{continuationTargets.map((target: Json) => <option key={target.id} value={target.id} disabled={!target.available}>{target.label}{target.available ? '' : ' · offline'}</option>)}</select><small>Requires image upload support because the previous ending frame anchors the next clip.</small></label>
      <label><span>Optional creative post-upscale</span><select value={value.postUpscale} onChange={(event) => select('postUpscale', event.target.value)}>{upscaleTargets.map((target: Json) => <option key={target.id} value={target.id} disabled={!target.available}>{target.label}{target.available ? '' : ` · ${target.unavailable_reason || 'offline'}`}</option>)}</select><small>The selected capable target runs LTX 2.3 Pixel Spatial as an isolated GGUF pass after approval. Delivery is capped at 1536px; longer videos are automatically split into overlapping 121-frame passes.</small></label>
      </div>
      <div className="target-grid">{targets.map((target: Json) => <article key={target.id} className={target.available ? 'target-card ready' : 'target-card'}><div><i /><span>{target.available ? 'Ready' : 'Setup needed'}</span></div><h3>{target.label}</h3><p>{target.description}</p><small>{target.capabilities?.map((capability: string) => capability.replace(/-/g, ' ')).join(' · ')}</small>{!target.available && <em>{target.unavailable_reason}</em>}</article>)}</div>
      <div className="execution-footer"><p>{info?.policy || 'Restart the API after changing machine configuration; running work is never migrated mid-render.'}</p><div><button className="button secondary" onClick={onRefresh}>Recheck machines</button><button className="button primary" onClick={onClose}>Done</button></div></div>
    </section>
  </div>;
}

function GpuMeter({ label, value, tone, detail }: { label: string; value: number; tone: string; detail?: string }) {
  const safe = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
  return <div className="gpu-meter"><div><span>{label}</span><strong>{detail || `${Math.round(safe)}%`}</strong></div><i><b className={`tone-${tone}`} style={{ width: `${safe}%` }} /></i></div>;
}

function NewProject({ backendFeatures, onClose, onCreated }: { backendFeatures: string[]; onClose: () => void; onCreated: (project: Json) => void }) {
  const [draft] = useState<Json | null>(() => savedProjectDraft());
  const [form, setForm] = useState(draft?.form || { title: '', topic: '', platform: 'reels', aspect_ratio: '9:16', duration_seconds: 15, style: 'cinematic, vivid, high-energy', prompt_mode: 'manual' as PromptMode });
  const [sourceKind, setSourceKind] = useState<'concept' | 'script'>(draft?.sourceKind === 'script' ? 'script' : 'concept');
  const [script, setScript] = useState(draft?.script || '');
  const [scriptFile, setScriptFile] = useState(draft?.scriptFile || '');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const scriptSupported = backendFeatures.includes('script-to-video');

  useEffect(() => {
    localStorage.setItem(NEW_PROJECT_DRAFT_KEY, JSON.stringify({ form, sourceKind, script, scriptFile, savedAt: Date.now() }));
  }, [form, sourceKind, script, scriptFile]);

  async function submit(event: FormEvent) {
    event.preventDefault(); setSaving(true); setError('');
    try {
      if (sourceKind === 'script' && !scriptSupported) {
        throw new Error('The frontend is newer than the running API. Restart the VidBangerGen API, then reopen this dialog. Your script is fine.');
      }
      const cleanScript = script.trim();
      const topic = sourceKind === 'script'
        ? (form.topic.trim() || cleanScript.replace(/\s+/g, ' ').slice(0, 500))
        : form.topic;
      const brief = {
        ...form,
        topic,
        source_kind: sourceKind,
        script: sourceKind === 'script' ? cleanScript : '',
        prompt_mode: sourceKind === 'script' ? 'assisted' : form.prompt_mode,
      };
      const value = await request('/projects', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ brief }) });
      if (sourceKind === 'script') {
        await request(`/projects/${value.id}/plan`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ concept_count: 1 }),
        });
      }
      localStorage.removeItem(NEW_PROJECT_DRAFT_KEY);
      onCreated(value);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not create project'); }
    finally { setSaving(false); }
  }

  return <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <form className="modal new-production-modal" onSubmit={submit}>
      <div className="modal-head"><div><span className="eyebrow">New production</span><h2>{sourceKind === 'script' ? 'Turn a script into a video' : 'Start with a clear brief'}</h2></div><button type="button" className="icon-button" onClick={onClose}>×</button></div>
      <div className="creation-mode">
        <button type="button" className={sourceKind === 'concept' ? 'active' : ''} onClick={() => setSourceKind('concept')}><strong>Start with a concept</strong><span>Build manually or ask Bonsai for creative directions.</span></button>
        <button type="button" className={sourceKind === 'script' ? 'active' : ''} onClick={() => { setSourceKind('script'); setForm({ ...form, prompt_mode: 'assisted' }); }}><strong>Script to video</strong><span>{scriptSupported ? 'Import a .txt screenplay or paste your complete script.' : 'API restart required · your running backend is older than this UI.'}</span></button>
      </div>
      <label>Project title<input autoFocus required value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} placeholder="Zero-G sneaker launch" /></label>
      {sourceKind === 'concept' ? <label>What should the video communicate?<textarea required rows={4} value={form.topic} onChange={(event) => setForm({ ...form, topic: event.target.value })} placeholder="A chrome sneaker assembles itself in zero gravity, then lands on the beat…" /></label> : <div className="script-import">
        <label className="script-file">Import plain text script<input type="file" accept=".txt,text/plain" onChange={async (event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          const text = await file.text();
          setScript(text); setScriptFile(file.name);
          if (!form.title.trim()) setForm({ ...form, title: file.name.replace(/\.txt$/i, '') });
        }} />{scriptFile && <span>{scriptFile} · {script.length.toLocaleString()} characters loaded</span>}</label>
        <label>Script<textarea required rows={12} value={script} onChange={(event) => { setScript(event.target.value); setScriptFile(''); }} placeholder={'INT. WORKSHOP - NIGHT\n\nMARA tightens the final bolt on a chrome bird. Its eyes ignite.\n\nMARA\nLet’s see if you remember how to fly.'} /></label>
        <label>Story intent or non-negotiables <span className="optional-label">optional</span><textarea rows={2} value={form.topic} onChange={(event) => setForm({ ...form, topic: event.target.value })} placeholder="Keep the ending intact, preserve all dialogue, emphasize the mother-daughter relationship…" /></label>
      </div>}
      <div className="form-grid four">
        <label>Platform<select value={form.platform} onChange={(event) => { const platform = event.target.value; setForm({ ...form, platform, aspect_ratio: platform === 'youtube' || platform === 'x' ? '16:9' : '9:16' }); }}><option value="reels">Instagram Reels</option><option value="tiktok">TikTok</option><option value="shorts">YouTube Shorts</option><option value="youtube">YouTube</option><option value="x">X</option></select></label>
        <label>Aspect ratio<select value={form.aspect_ratio} onChange={(event) => setForm({ ...form, aspect_ratio: event.target.value })}><option value="9:16">Vertical · 9:16</option><option value="16:9">Landscape · 16:9</option><option value="1:1">Square · 1:1</option></select></label>
        <label>Target runtime<select value={form.duration_seconds} onChange={(event) => setForm({ ...form, duration_seconds: Number(event.target.value) })}><option value={5}>5 seconds</option><option value={10}>10 seconds</option><option value={15}>15 seconds</option><option value={30}>30 seconds</option><option value={60}>60 seconds</option><option value={90}>90 seconds</option><option value={120}>120 seconds</option></select></label>
        {sourceKind === 'concept' ? <label>Prompt author<select value={form.prompt_mode} onChange={(event) => setForm({ ...form, prompt_mode: event.target.value as PromptMode })}><option value="manual">Manual / human</option><option value="assisted">Bonsai 27B</option></select></label> : <div className="script-director-setting"><span>Breakdown</span><strong>Bonsai 27B + editable review</strong></div>}
      </div>
      <label>Visual direction<input value={form.style} onChange={(event) => setForm({ ...form, style: event.target.value })} /></label>
      {sourceKind === 'script' && !scriptSupported && <p className="api-version-warning"><strong>Backend restart needed</strong><span>Stop the local VidBangerGen dev process and run <code>npm run dev</code> again. Do not restart ComfyUI.</span></p>}
      {sourceKind === 'script' && script.trim() && <p className="local-draft-note">Saved in this browser · {script.length.toLocaleString()} characters</p>}
      {error && <p className="inline-error">{error}</p>}
      <div className="modal-actions"><button type="button" className="button secondary" onClick={onClose}>Cancel</button><button className="button primary" disabled={saving || (sourceKind === 'script' && (!scriptSupported || script.trim().length < 20))}>{saving ? (sourceKind === 'script' ? 'Breaking down script…' : 'Creating…') : (sourceKind === 'script' ? 'Create script storyboard' : 'Create project')}</button></div>
    </form>
  </div>;
}

function ProjectStudio({ project, models, creativeLab, execution, executionInfo, onExecutionChange, busy, refresh, act }: { project: Json; models: Json | null; creativeLab: Json | null; execution: ExecutionPreferences; executionInfo: Json | null; onExecutionChange: (value: ExecutionPreferences) => void; busy: string; refresh: () => Promise<void>; act: (label: string, operation: () => Promise<any>, success?: string) => Promise<any> }) {
  const concepts = project.concepts || [];
  const selectedConcept = concepts.find((item: Json) => item.selected) || concepts[0];
  const shots = selectedConcept?.shots || [];
  const [candidateCount, setCandidateCount] = useState(2);
  const [estimate, setEstimate] = useState<Json | null>(null);
  const [assetKind, setAssetKind] = useState('image');
  const [assetFile, setAssetFile] = useState<File | null>(null);
  const [assetLabel, setAssetLabel] = useState('');
  const [assetNotes, setAssetNotes] = useState('');
  const [referenceAsset, setReferenceAsset] = useState('');
  const [referenceMode, setReferenceMode] = useState<'first-shot' | 'every-shot'>('first-shot');
  const [referenceEngine, setReferenceEngine] = useState<'union' | 'ingredients'>('union');
  const [referenceDescription, setReferenceDescription] = useState('');
  const [audioMode, setAudioMode] = useState<StudioAudioPolicy>('shot');
  const [brandFile, setBrandFile] = useState<File | null>(null);
  const [brandRole, setBrandRole] = useState('logo');
  const [brandLabel, setBrandLabel] = useState('');
  const [brandNotes, setBrandNotes] = useState('');
  const [showJobs, setShowJobs] = useState(false);
  const activeJobs = (project.jobs || []).filter((job: Json) => ACTIVE_JOBS.has(job.status));
  const failedJobs = (project.jobs || []).filter((job: Json) => job.status === 'failed' || job.status === 'cancelled');
  const upscaleJobs = (project.jobs || []).filter((job: Json) => job.kind === 'upscale');
  const transformJobs = (project.jobs || []).filter((job: Json) => job.kind === 'creative_transform');
  const readyTransformModes = (creativeLab?.modes || []).filter(
    (mode: Json) => mode.ready && Object.prototype.hasOwnProperty.call(TRANSFORM_UI, mode.id)
  );
  const ingredientsReady = Boolean(creativeLab?.modes?.find((mode: Json) => mode.id === 'ingredients')?.ready);
  const lipdubReady = Boolean(creativeLab?.companion_modes?.find((mode: Json) => mode.id === 'lipdub')?.ready);
  const scriptMode = project.brief.source_kind === 'script';
  const elements = selectedConcept?.data?.elements || [];
  const visualAssets = (project.assets || []).filter((asset: Json) =>
    ['image', 'reference_sheet', 'brand'].includes(asset.kind) && String(asset.mime_type || '').startsWith('image/')
  );
  const maskAssets = (project.assets || []).filter((asset: Json) =>
    String(asset.mime_type || '').startsWith('image/') || String(asset.mime_type || '').startsWith('video/')
  );
  const brandAssets = visualAssets.filter((asset: Json) => asset.kind === 'brand');
  const referenceSheets = visualAssets.filter((asset: Json) => asset.kind === 'reference_sheet');
  const scoringEnabled = Boolean(models?.ollama?.visual_scoring_enabled);
  const selectedDrafts = shots.filter((shot: Json) => shot.selected_candidate_id).length;
  const selectedFinals = shots.filter((shot: Json) => {
    const selected = (shot.candidates || []).find((candidate: Json) => candidate.id === shot.selected_candidate_id);
    return selected && !selected.draft;
  }).length;
  const ingredientsTooLong = shots.some((shot: Json) => Number(shot.duration_seconds) > 5);
  const ingredientsBlocked = referenceEngine === 'ingredients' && (
    !ingredientsReady || !referenceAsset || referenceDescription.trim().length < 10 || ingredientsTooLong
  );

  useEffect(() => {
    setAudioMode('shot');
    setReferenceEngine('union');
    setReferenceDescription('');
  }, [project.id]);

  useEffect(() => {
    if (referenceEngine !== 'ingredients') return;
    setAudioMode('silent');
    setReferenceMode('every-shot');
    const selected = referenceSheets.find((asset: Json) => asset.id === referenceAsset);
    if (!selected) {
      setReferenceAsset('');
      setReferenceDescription('');
    } else if (!referenceDescription.trim() && selected.metadata?.notes) {
      setReferenceDescription(String(selected.metadata.notes));
    }
  }, [referenceEngine, referenceAsset, referenceSheets.length]);

  useEffect(() => {
    if (!selectedConcept || !shots.length) { setEstimate(null); return; }
    request(`/projects/${project.id}/generation-estimate?candidates_per_shot=${candidateCount}&execution_target=${encodeURIComponent(execution.generation)}&reference_engine=${referenceEngine}`).then(setEstimate).catch(() => setEstimate(null));
  }, [project.id, selectedConcept?.id, shots.length, candidateCount, execution.generation, referenceEngine]);

  async function generate() {
    const aspect = project.brief.aspect_ratio;
    const sizes: Record<string, [number, number]> = { '9:16': [256, 448], '16:9': [448, 256], '1:1': [320, 320] };
    const [width, height] = sizes[aspect] || sizes['9:16'];
    const settings = { width, height, profile: 'motion-draft-4x3', audio_mode: referenceEngine === 'ingredients' ? 'silent' : audioMode, execution_target: execution.generation, ...(referenceEngine === 'ingredients' ? { reference_engine: 'ingredients' } : {}), ...(referenceAsset ? { reference_image_asset_id: referenceAsset, reference_mode: referenceEngine === 'ingredients' ? 'every-shot' : referenceMode } : {}), ...(referenceEngine === 'ingredients' ? { reference_sheet_description: referenceDescription.trim(), ingredients_strength: 1 } : {}) };
    await act('generate', async () => {
      const queue = (generationSettings: Json) => request(`/projects/${project.id}/generate`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ concept_id: selectedConcept.id, candidates_per_shot: candidateCount, settings: generationSettings }) });
      try {
        return await queue(settings);
      } catch (cause) {
        const failure = cause as Error & { status?: number };
        // A Vite tab can be newer than a non-reloading API process. The old API
        // rejects reference_engine before creating jobs, so the ordinary Union
        // path can safely retry with the legacy payload. Ingredients must never
        // degrade to Union conditioning without telling the operator.
        if (referenceEngine === 'union' && failure.status === 422 && failure.message.includes('reference_engine')) {
          const legacySettings: Json = { ...settings };
          delete legacySettings.reference_engine;
          return queue(legacySettings);
        }
        throw cause;
      }
    }, referenceEngine === 'ingredients' ? 'Ingredients generation queued' : 'Draft generation queued');
  }

  async function uploadAsset(event: FormEvent) {
    event.preventDefault();
    if (!assetFile) return;
    const data = new FormData(); data.append('kind', assetKind); data.append('file', assetFile);
    if (assetLabel.trim()) data.append('label', assetLabel.trim());
    if (assetNotes.trim()) data.append('notes', assetNotes.trim());
    await act('asset', () => request(`/projects/${project.id}/assets`, { method: 'POST', body: data }), 'Asset added');
    setAssetFile(null); setAssetLabel(''); setAssetNotes('');
  }

  async function uploadBrandAsset(event: FormEvent) {
    event.preventDefault();
    if (!brandFile) return;
    const data = new FormData();
    data.append('kind', 'brand'); data.append('brand_role', brandRole); data.append('file', brandFile);
    if (brandLabel.trim()) data.append('label', brandLabel.trim());
    if (brandNotes.trim()) data.append('notes', brandNotes.trim());
    await act('brand-asset', () => request(`/projects/${project.id}/assets`, { method: 'POST', body: data }), 'Brand asset added');
    setBrandFile(null); setBrandLabel(''); setBrandNotes('');
  }

  return <>
    <section className="project-header">
      <div><span className="eyebrow">{scriptMode ? 'Script to video · ' : ''}{project.brief.platform} · {project.brief.aspect_ratio} · {project.brief.duration_seconds}s</span><div className="title-line"><h1>{project.title}</h1><Pill status={project.status} /></div><p>{scriptMode ? `${project.brief.script?.length?.toLocaleString() || 0} script characters imported · Review every generated shot before rendering.` : project.brief.topic}</p></div>
      <div className="header-actions"><button className="button secondary" onClick={() => refresh()}>Refresh</button><button className="button secondary" onClick={() => setShowJobs(!showJobs)}>Jobs {activeJobs.length ? <b className="count">{activeJobs.length}</b> : null}</button></div>
    </section>

    {showJobs && <JobDrawer jobs={project.jobs || []} busy={busy} act={act} />}

    <div className="stage-strip">
      <Stage number="1" label="Direct" done={Boolean(concepts.length)} active={!concepts.length} />
      <Stage number="2" label="Storyboard" done={Boolean(shots.length)} active={Boolean(concepts.length && !shots.length)} />
      <Stage number="3" label="Review" done={Boolean(shots.length && selectedDrafts === shots.length)} active={Boolean(shots.length && selectedDrafts < shots.length)} />
      <Stage number="4" label="Finish" done={Boolean(shots.length && selectedFinals === shots.length)} active={Boolean(shots.length && selectedDrafts === shots.length)} />
    </div>

    {!concepts.length ? (
      <PlanningPanel project={project} model={models?.ollama?.director || 'bonsai-27b:latest'} busy={busy} act={act} />
    ) : (
      <>
        <section className="panel concept-panel">
          <div className="section-head"><div><span className="eyebrow">{scriptMode ? 'Script breakdown' : 'Creative direction'}</span><h2>{scriptMode ? 'Review the adapted narrative' : 'Choose the strongest concept'}</h2></div><span className="model-chip">{project.brief.prompt_mode === 'manual' ? 'Human-authored · exact prompts' : `${models?.ollama?.director || 'Bonsai 27B'} assisted`}</span></div>
          <div className="concept-grid">
            {concepts.map((concept: Json) => <button key={concept.id} className={`concept-card ${concept.id === selectedConcept.id ? 'selected' : ''}`} onClick={() => act('concept', () => request(`/projects/${project.id}/concepts/${concept.id}/select`, { method: 'POST' }))}>
              <div><span className="concept-number">0{concept.position + 1}</span>{concept.selected && <span className="selected-label">Selected</span>}</div>
              <h3>{concept.title}</h3><p className="concept-hook">{concept.hook}</p><p>{concept.treatment}</p>
            </button>)}
          </div>
        </section>

        {scriptMode && <section className="panel elements-panel">
          <div className="section-head"><div><span className="eyebrow">Continuity bible</span><h2>Reusable Elements</h2><p>Characters, locations, and important objects extracted from the script. Their descriptions are carried into the storyboard for cross-shot consistency.</p></div><span className="model-chip">{elements.length} extracted</span></div>
          {elements.length ? <div className="element-grid">{elements.map((element: Json, index: number) => <article className={`element-card element-${element.type}`} key={`${element.type}-${element.name}-${index}`}><span>{element.type}</span><h3>{element.name}</h3><p>{element.description || 'Reusable continuity element from the imported script.'}</p></article>)}</div> : <div className="manual-banner"><strong>No explicit Elements found</strong><span>You can still edit each shot prompt and attach reference sheets below before generation.</span></div>}
        </section>}

        <section className="panel">
          <div className="section-head"><div><span className="eyebrow">{scriptMode ? 'Script storyboard' : 'Storyboard'}</span><h2>{shots.length} shots · {shots.reduce((sum: number, shot: Json) => sum + Number(shot.duration_seconds), 0)} seconds</h2></div><button className="button ghost" onClick={() => {
            if (window.confirm('Replace the current plan? Existing candidates for this plan will be removed.')) {
              act('replan', () => request(`/projects/${project.id}/plan`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ concept_count: scriptMode ? 1 : 3, regenerate: true }) }), scriptMode ? 'Script breakdown queued' : 'Bonsai replan queued');
            }
          }}>{scriptMode ? 'Rebuild script breakdown' : 'Rebuild with Bonsai'}</button></div>
          <div className="shot-list">
            {shots.map((shot: Json, index: number) => <ShotCard key={shot.id} shot={shot} index={index} visualAssets={visualAssets} busy={busy} act={act} />)}
          </div>
        </section>

        <section className="panel brand-kit-panel">
          <div className="section-head"><div><span className="eyebrow">Reusable identity</span><h2>Brand & product kit</h2><p>Upload clean logos, products, wardrobe, signs, people, and locations once. These assets can anchor the whole draft set or a specific storyboard shot.</p></div><span className="model-chip">Union IC-LoRA · GGUF</span></div>
          <form className="brand-upload" onSubmit={uploadBrandAsset}>
            <label>Asset role<select value={brandRole} onChange={(event) => setBrandRole(event.target.value)}><option value="logo">Logo / emblem</option><option value="product">Product</option><option value="character">Person / character</option><option value="wardrobe">Wardrobe</option><option value="sign">Sign / packaging</option><option value="location">Location</option><option value="style_reference">Look / style reference</option></select></label>
            <label>Name<input value={brandLabel} onChange={(event) => setBrandLabel(event.target.value)} placeholder="Forest-green roundel" /></label>
            <label className="brand-file">PNG, JPG, or WebP<input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => setBrandFile(event.target.files?.[0] || null)} /><span>{brandFile?.name || 'Choose a clean reference image'}</span></label>
            <label className="brand-notes">Non-negotiable details<input value={brandNotes} onChange={(event) => setBrandNotes(event.target.value)} placeholder="Keep the white leaf, circular border, and forest-green color unchanged" /></label>
            <button className="button secondary" disabled={!brandFile || busy === 'brand-asset'}>{busy === 'brand-asset' ? 'Uploading…' : 'Add to Brand Kit'}</button>
          </form>
          {brandAssets.length ? <div className="brand-grid">{brandAssets.map((asset: Json) => <article className="brand-card" key={asset.id}><img src={`${API}/assets/${asset.id}`} alt={asset.metadata?.label || asset.filename} /><div><span>{String(asset.metadata?.brand_role || 'reference').replace('_', ' ')}</span><strong>{asset.metadata?.label || asset.filename}</strong>{asset.metadata?.notes && <p>{asset.metadata.notes}</p>}</div></article>)}</div> : <div className="brand-empty"><strong>No reusable identity assets yet</strong><span>For clothing or signage, upload a complete mockup with the mark already placed—not only a floating logo.</span></div>}
          <div className="brand-guidance"><strong>Prompt it visibly</strong><span>Describe where the reference belongs: “preserve the supplied emblem exactly on the front of the forest-green cap.” LTX can preserve the look, but generated lettering may still drift.</span></div>
        </section>

        <section className="panel generation-panel">
          <div className="section-head"><div><span className="eyebrow">Generation lab</span><h2>Generate candidates</h2><p>{referenceEngine === 'ingredients' ? 'Generate identity-consistent production takes from a visual bible with the official Ingredients conditioning pattern.' : 'Low-resolution first for stronger motion. Every draft stays on the stable LTX 2.3 GGUF path.'}</p></div></div>
          <div className="generation-controls">
            <label>Generation machine<select value={execution.generation} onChange={(event) => onExecutionChange({ ...execution, generation: event.target.value })}><option value="auto">Auto · healthy LTX pool</option>{(executionInfo?.targets || []).filter((target: Json) => target.capabilities?.includes('ltx-generation')).map((target: Json) => <option key={target.id} value={target.id} disabled={!target.available}>{target.label}{target.available ? '' : ' · offline'}</option>)}</select></label>
            <label>Variations per shot<select value={candidateCount} onChange={(event) => setCandidateCount(Number(event.target.value))}><option value={1}>1 · quickest test</option><option value={2}>2 · balanced</option><option value={4}>4 · best-of-N</option><option value={6}>6 · wide search</option></select></label>
            <label id="ingredients-engine">Reference workflow<select value={referenceEngine} onChange={(event) => setReferenceEngine(event.target.value as 'union' | 'ingredients')}><option value="union">Stable Union anchor · fast drafts</option><option value="ingredients" disabled={!ingredientsReady}>Ingredients visual bible{ingredientsReady ? ' · production consistency' : ' · unavailable'}</option></select></label>
            <label>Visual identity anchor<select value={referenceAsset} onChange={(event) => { const value = event.target.value; setReferenceAsset(value); const selected = visualAssets.find((asset: Json) => asset.id === value); if (referenceEngine === 'ingredients') setReferenceDescription(String(selected?.metadata?.notes || '')); }}><option value="">{referenceEngine === 'ingredients' ? 'Choose a visual bible…' : 'None · text to video'}</option>{(referenceEngine === 'ingredients' ? referenceSheets : visualAssets).map((asset: Json) => <option key={asset.id} value={asset.id}>{asset.metadata?.label || asset.filename} · {String(asset.metadata?.brand_role || asset.kind).replace('_', ' ')}</option>)}</select></label>
            <label>Apply anchor<select value={referenceMode} disabled={!referenceAsset || referenceEngine === 'ingredients'} onChange={(event) => setReferenceMode(event.target.value as 'first-shot' | 'every-shot')}><option value="first-shot">Opening / first selected shot</option><option value="every-shot">Every shot · strongest consistency</option></select></label>
            <label>Generated audio<select value={referenceEngine === 'ingredients' ? 'silent' : audioMode} disabled={referenceEngine === 'ingredients'} onChange={(event) => setAudioMode(event.target.value as StudioAudioPolicy)}><option value="shot">Use each shot’s audio intent · recommended</option><option value="ambient">Force ambience + Foley · no speech</option><option value="silent">Force silent · guaranteed no voice</option><option value="native-dialogue">Only enabled shot dialogue · generative</option></select></label>
            {referenceEngine === 'ingredients' && <div className="ingredients-contract"><strong>Official Ingredients contract</strong><span>768×448 · 121 frames · up to 5 seconds · every shot uses the full static reference sheet · silent output</span><label>Describe every panel<textarea rows={4} value={referenceDescription} onChange={(event) => setReferenceDescription(event.target.value)} placeholder="Character: front-facing close-up and full-body turnaround… Product: exact bottle, label colors, and cap… Location: clean store aisle…" /></label>{ingredientsTooLong && <em>One or more shots exceed 5 seconds. Split them into shorter beats before using Ingredients.</em>}</div>}
            <div className="estimate"><strong>{estimate ? `~${estimate.estimated_wall_minutes} min` : 'Calculating…'}</strong><span>{estimate ? `${estimate.render_count} renders across ${estimate.configured_workers} worker${estimate.configured_workers === 1 ? '' : 's'}` : 'Worker estimate'}</span></div>
            <button className="button primary large" disabled={Boolean(busy) || !shots.length || ingredientsBlocked} onClick={generate}>{busy === 'generate' ? 'Checking workers…' : referenceEngine === 'ingredients' ? 'Generate with Ingredients' : 'Generate draft set'}</button>
          </div>
        </section>

        {shots.some((shot: Json) => shot.candidates?.length) && <section className="review-section" id="candidate-review">
          <div className="section-head"><div><span className="eyebrow">Human review</span><h2>Pick the keeper for every shot</h2><p>{scoringEnabled ? 'Automated scores are guidance. Your selection remains authoritative.' : 'Vision scoring is off. Compare the actual motion, identity, composition, and ending frame yourself.'}</p></div><span className="review-count">{selectedDrafts}/{shots.length} selected</span></div>
          {shots.map((shot: Json, index: number) => shot.candidates?.length ? <CandidateRow key={shot.id} shot={shot} index={index} scoringEnabled={scoringEnabled} upscaleTarget={execution.postUpscale} upscaleJobs={upscaleJobs} transformJobs={transformJobs} transformModes={readyTransformModes} inOutpaintingReady={Boolean(creativeLab?.modes?.find((mode: Json) => mode.id === 'in-outpainting')?.ready)} lipdubReady={lipdubReady} maskAssets={maskAssets} executionInfo={executionInfo} busy={busy} act={act} /> : null)}
        </section>}

        <EnhancementLab creativeLab={creativeLab} execution={execution} executionInfo={executionInfo} project={project} act={act} hasReviewableClip={shots.some((shot: Json) => shot.candidates?.some((candidate: Json) => candidate.artifact))} onUseIngredients={() => { setReferenceEngine('ingredients'); document.querySelector('.generation-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }} />

        <FinishPanel project={project} shots={shots} selectedDrafts={selectedDrafts} selectedFinals={selectedFinals} execution={execution} executionInfo={executionInfo} busy={busy} act={act} />
      </>
    )}

    <section className="panel asset-panel">
      <div className="section-head"><div><span className="eyebrow">Production assets</span><h2>References, audio, and visual bibles</h2></div></div>
      <form className={`asset-upload ${assetKind === 'reference_sheet' ? 'visual-bible-upload' : ''}`} onSubmit={uploadAsset}><select value={assetKind} onChange={(event) => setAssetKind(event.target.value)}><option value="image">Image / static mask</option><option value="video">Video / animated mask</option><option value="reference_sheet">Visual bible</option><option value="reference">Audio reference</option><option value="music">Music</option><option value="voiceover">Voiceover</option></select><input type="file" accept={assetKind === 'reference_sheet' || assetKind === 'image' ? 'image/png,image/jpeg,image/webp' : assetKind === 'video' ? 'video/mp4,video/webm,video/quicktime' : undefined} onChange={(event) => setAssetFile(event.target.files?.[0] || null)} />{assetKind === 'reference_sheet' && <><input value={assetLabel} onChange={(event) => setAssetLabel(event.target.value)} placeholder="Visual bible name" /><textarea rows={3} value={assetNotes} onChange={(event) => setAssetNotes(event.target.value)} placeholder="Describe every panel: character close-up and turnaround, product, wardrobe, logo/sign, and clean location…" /></>}<button className="button secondary" disabled={!assetFile || busy === 'asset' || (assetKind === 'reference_sheet' && assetNotes.trim().length < 10)}>{busy === 'asset' ? 'Uploading…' : 'Add asset'}</button></form>
      {assetKind === 'reference_sheet' && <p className="visual-bible-guidance">Use one clean panel per element on a black background, with no text labels. Give the most important character or product the largest panels.</p>}
      <div className="asset-list">{(project.assets || []).map((asset: Json) => <a key={asset.id} href={`${API}/assets/${asset.id}`}><span>{asset.kind.replace('_', ' ')}</span><strong>{asset.filename}</strong></a>)}{!project.assets?.length && <p>No project assets uploaded.</p>}</div>
    </section>
  </>;
}

function EnhancementLab({ creativeLab, execution, executionInfo, project, act, hasReviewableClip, onUseIngredients }: { creativeLab: Json | null; execution: ExecutionPreferences; executionInfo: Json | null; project: Json; act: (label: string, operation: () => Promise<any>, success?: string) => Promise<any>; hasReviewableClip: boolean; onUseIngredients: () => void }) {
  const active = creativeLab?.active_pipeline;
  const modes = creativeLab?.modes || [];
  const companions = creativeLab?.companion_modes || [];
  const [category, setCategory] = useState('all');
  const categories: string[] = [
    'all', ...Array.from(
      new Set<string>(modes.map((mode: Json) => String(mode.category || 'other')))
    ),
  ];
  const visibleModes = category === 'all' ? modes : modes.filter((mode: Json) => mode.category === category);
  function chooseMode(mode: Json) {
    if (!mode.ready) return;
    if (mode.id === 'ingredients') { onUseIngredients(); return; }
    if (mode.id === 'cinemagraph') {
      document.getElementById('cinemagraph-tool')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }
    if (mode.id === 'in-outpainting') {
      document.getElementById('candidate-review')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    if (!(mode.id === 'pixel-spatial-upscaler' || TRANSFORM_UI[mode.id as TransformMode]) || !hasReviewableClip) return;
    document.getElementById('candidate-review')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  return <section className="panel enhancement-lab">
    <div className="section-head"><div><span className="eyebrow">LTX Creation Lab</span><h2>Motion first, detail after approval</h2><p>The stable path stays untouched. Optional transformations run as separate passes only after you like the source clip.</p></div><a className="model-chip lab-source" href={creativeLab?.collection_url} target="_blank" rel="noreferrer">Official collection ↗</a></div>
    <div className="active-upscale"><div className="upscale-icon">2×</div><div><span>Active in every render · {executionLabel(executionInfo, execution.generation)}</span><h3>{active?.name || 'Motion-first latent upscale'}</h3><p>{active ? `${active.generation_steps}-step motion generation → ${active.upscale_steps}-step LTX latent upscale using the Q4_K_M GGUF.` : 'The current GGUF workflow generates motion at low resolution and performs its built-in detail pass.'}</p></div><strong>Ready now</strong></div>
    <div className="lab-filters" aria-label="Creation Lab categories">{categories.map((value) => <button key={value} className={category === value ? 'active' : ''} onClick={() => setCategory(value)}>{value.replace(/-/g, ' ')}</button>)}</div>
    <div className="enhancement-grid">{visibleModes.map((mode: Json) => {
      const recommended = mode.id === creativeLab?.recommended_next_mode;
      const activeAction = (['ingredients', 'pixel-spatial-upscaler', 'cinemagraph', 'in-outpainting'].includes(mode.id) || Boolean(TRANSFORM_UI[mode.id as TransformMode])) && mode.ready;
      const needsClip = !['ingredients', 'cinemagraph'].includes(mode.id);
      const actionLabel = mode.id === 'ingredients' && activeAction ? 'Use in Generation Lab' : mode.id === 'cinemagraph' && activeAction ? 'Open image tool' : activeAction ? (hasReviewableClip ? 'Apply from a clip card' : 'Generate a clip first') : mode.installed ? 'Model installed · workflow pending' : mode.readiness_state === 'inventory_unavailable' ? 'Inventory unavailable' : 'Model setup required';
      const readinessLabel = mode.ready ? 'Ready' : mode.installed ? 'Installed · workflow pending' : mode.readiness_state === 'inventory_unavailable' ? 'Inventory unknown' : 'Model missing';
      return <article className={`enhancement-card ${recommended ? 'recommended' : ''}`} key={mode.id}>
        <div><span>{String(mode.category || 'mode').replace('_', ' ')}</span><em className={mode.ready ? 'ready' : mode.installed ? 'installed' : ''}>{readinessLabel}</em></div>
        <h3>{mode.name}</h3><small className="lab-input">Input · {mode.input_kind || 'workflow-specific source'}</small><p>{mode.purpose}</p>
        {recommended && <strong className="recommended-label">Recommended next</strong>}
        <div className="enhancement-actions"><button className="button secondary" disabled={!activeAction || (needsClip && !hasReviewableClip)} onClick={() => chooseMode(mode)}>{actionLabel}</button>{mode.model_url && <a href={mode.model_url} target="_blank" rel="noreferrer">Model card ↗</a>}</div>
      </article>;
    })}</div>
    <CinemagraphTool ready={Boolean(modes.find((mode: Json) => mode.id === 'cinemagraph')?.ready)} assets={project.assets || []} jobs={project.jobs || []} targetId={execution.postUpscale} executionInfo={executionInfo} act={act} />
    <div className="lab-inventory"><span>Remote model inventory</span><strong>{creativeLab?.model_inventory?.observed ? `${creativeLab.model_inventory.file_count} LoRA files observed` : 'Read-only inventory unavailable'}</strong><small>Installed weights and runnable workflows are tracked separately. Inventory checks never install, move, or delete server files.</small></div>
    {companions.length > 0 && <div className="companion-modes"><span>LTX 2.3 companion workflows</span>{companions.map((mode: Json) => <a key={mode.id} href={mode.model_url} target="_blank" rel="noreferrer"><strong>{mode.name}</strong><small>{mode.ready ? 'Ready' : mode.installed ? 'Model installed · workflow pending' : 'Model setup required'} · {mode.input_kind}</small></a>)}</div>}
    <div className="post-upscale-route"><span>Optional Pixel Spatial route</span><strong>{executionLabel(executionInfo, execution.postUpscale, 'Not configured')}</strong><small>Source audio is preserved. Long clips are rendered as seamless 121-frame passes; 1536px delivery cap.</small></div>
    <p className="enhancement-policy">These controls are capability-gated. Enabling a card requires the matching model and audited workflow on your configured worker; VidBangerGen will never silently fall back to FP8 or claim an unavailable pass ran.</p>
  </section>;
}

function CinemagraphTool({ ready, assets, jobs, targetId, executionInfo, act }: { ready: boolean; assets: Json[]; jobs: Json[]; targetId: string; executionInfo: Json | null; act: (label: string, operation: () => Promise<any>, success?: string) => Promise<any> }) {
  const images = assets.filter((asset: Json) => String(asset.mime_type || '').startsWith('image/'));
  const existingJob = jobs.find((value: Json) => value.kind === 'cinemagraph');
  const [assetId, setAssetId] = useState(images[0]?.id || '');
  const [prompt, setPrompt] = useState('');
  const [strength, setStrength] = useState(1);
  const [job, setJob] = useState<Json | null>(existingJob || null);
  const [error, setError] = useState('');
  const target = executionInfo?.targets?.find((value: Json) => value.id === targetId);
  const active = Boolean(job && ACTIVE_JOBS.has(job.status));

  useEffect(() => {
    if (!assetId && images[0]?.id) setAssetId(images[0].id);
  }, [assetId, images.length]);
  useEffect(() => {
    if (existingJob && (!job || existingJob.id === job.id)) setJob(existingJob);
  }, [existingJob?.id, existingJob?.status, existingJob?.progress]);
  useEffect(() => {
    if (!job?.id || !ACTIVE_JOBS.has(job.status)) return;
    const interval = window.setInterval(() => request(`/jobs/${job.id}`).then(setJob).catch((cause) => setError(cause.message)), 2000);
    return () => window.clearInterval(interval);
  }, [job?.id, job?.status]);

  async function start() {
    setError('');
    try {
      const value = await act('cinemagraph', () => request('/creative-lab/cinemagraph', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ asset_id: assetId, target_id: targetId, prompt: prompt.trim(), strength }),
      }), 'Cinemagraph queued');
      setJob(value.job);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Cinemagraph could not start');
    }
  }

  if (!ready) return null;
  return <div className="cinemagraph-tool" id="cinemagraph-tool">
    <div><span className="eyebrow">Selective motion · image to video</span><h3>Cinemagraph maker</h3><p>Pick a still and describe exactly one moving element. The camera, people, background, and everything else stay frozen.</p></div>
    <div className="cinemagraph-controls">
      <label>Source image<select value={assetId} onChange={(event) => setAssetId(event.target.value)}><option value="">Choose an uploaded image…</option>{images.map((asset: Json) => <option key={asset.id} value={asset.id}>{asset.metadata?.label || asset.filename}</option>)}</select></label>
      <label>Only this moves<input value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="only the neon sign subtly pulses and flickers" /></label>
      <label>Motion strength · {strength.toFixed(1)}<input type="range" min="0.7" max="3" step="0.1" value={strength} onChange={(event) => setStrength(Number(event.target.value))} /></label>
      {job?.status === 'succeeded' ? <a className="button success" href={`${API}/jobs/${job.id}/output`}>Download Cinemagraph</a> : <button className="button primary" disabled={!assetId || prompt.trim().length < 3 || !target?.available || active} onClick={start}>{active ? `Rendering · ${Math.round(Number(job?.progress || 0) * 100)}%` : 'Create 1-second loop'}</button>}
    </div>
    {!images.length && <small>Upload a JPG, PNG, or WebP in Production Assets below, then select it here.</small>}
    {job?.status === 'failed' && <span className="inline-job-error">{job.error || 'Cinemagraph failed'}</span>}
    {error && <span className="inline-job-error">{error}</span>}
  </div>;
}

function Stage({ number, label, done, active }: { number: string; label: string; done: boolean; active: boolean }) {
  return <div className={`stage ${done ? 'done' : ''} ${active ? 'active' : ''}`}><span>{done ? '✓' : number}</span><strong>{label}</strong></div>;
}

function PlanningPanel({ project, model, busy, act }: { project: Json; model: string; busy: string; act: (label: string, operation: () => Promise<any>, success?: string) => Promise<any> }) {
  const [mode, setMode] = useState<PromptMode>(project.brief.prompt_mode || 'manual');
  const [shots, setShots] = useState<Json[]>([{ title: 'Hero shot', purpose: 'hook', duration_seconds: project.brief.duration_seconds, prompt: '', negative_prompt: '', camera: '', audio: '', caption: '', transition: 'hard cut' }]);
  const total = shots.reduce((sum, shot) => sum + Number(shot.duration_seconds || 0), 0);
  const scriptMode = project.brief.source_kind === 'script';
  const scriptPlanning = (project.jobs || []).some((job: Json) => job.kind === 'creative_plan' && ACTIVE_JOBS.has(job.status));

  function updateShot(index: number, key: string, value: any) {
    setShots(shots.map((shot, position) => position === index ? { ...shot, [key]: value } : shot));
  }

  function addShot() {
    setShots([...shots, { title: `Shot ${shots.length + 1}`, purpose: 'build', duration_seconds: 5, prompt: '', negative_prompt: '', camera: '', audio: '', caption: '', transition: 'hard cut' }]);
  }

  if (scriptMode) {
    return <section className="panel direction-panel script-planning-panel">
      <div className="section-head"><div><span className="eyebrow">Stage 1 · Script breakdown</span><h2>{scriptPlanning ? 'Bonsai is building your storyboard' : 'Script ready for breakdown'}</h2><p>The script becomes editable scenes, shots, camera directions, sound notes, and recurring Elements before any GPU render begins.</p></div><span className="model-chip">{model}</span></div>
      <div className="script-preview"><div><span>Imported script</span><strong>{project.brief.script?.length?.toLocaleString() || 0} characters · {project.brief.duration_seconds}s target</strong></div><pre>{project.brief.script}</pre></div>
      {scriptPlanning ? <div className="script-planning-progress"><i /><span>Analyzing scenes, characters, locations, objects, dialogue, and shot timing…</span></div> : <button className="button primary large" disabled={Boolean(busy)} onClick={() => act('script-plan', () => request(`/projects/${project.id}/plan`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ concept_count: 1 }) }), 'Script breakdown queued')}>{busy === 'script-plan' ? 'Starting…' : 'Build editable storyboard'}</button>}
    </section>;
  }

  return <section className="panel direction-panel">
    <div className="section-head"><div><span className="eyebrow">Stage 1 · Direction</span><h2>Who writes the generation prompts?</h2></div></div>
    <div className="author-toggle">
      <button className={mode === 'manual' ? 'active' : ''} onClick={() => setMode('manual')}><strong>Manual override</strong><span>Your text reaches LTX exactly as written. Ollama is bypassed.</span></button>
      <button className={mode === 'assisted' ? 'active' : ''} onClick={() => setMode('assisted')}><strong>Bonsai 27B assisted</strong><span>{model} builds concepts and shot prompts from the brief.</span></button>
    </div>
    {mode === 'assisted' ? <div className="assisted-card"><div><span className="model-orb">27B</span><div><h3>Creative Director</h3><p>Bonsai produces three concepts with retention-aware, LTX-ready storyboards. You review before GPU work starts.</p></div></div><button className="button primary large" disabled={Boolean(busy)} onClick={() => act('plan', () => request(`/projects/${project.id}/plan`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ concept_count: 3 }) }), 'Bonsai planning queued')}>{busy === 'plan' ? 'Bonsai is directing…' : 'Build 3 concepts'}</button></div> : <>
      <div className="manual-banner"><strong>Human-in-the-loop is on</strong><span>No prompt expansion, strategy suffixes, or Ollama calls. Best-of-N varies only the seed.</span></div>
      <div className="manual-shots">
        {shots.map((shot, index) => <div className="manual-shot" key={index}>
          <div className="manual-shot-head"><span>Shot {index + 1}</span>{shots.length > 1 && <button onClick={() => setShots(shots.filter((_, position) => position !== index))}>Remove</button>}</div>
          <div className="form-grid three"><label>Title<input value={shot.title} onChange={(event) => updateShot(index, 'title', event.target.value)} /></label><label>Purpose<select value={shot.purpose} onChange={(event) => updateShot(index, 'purpose', event.target.value)}><option value="hook">Hook</option><option value="build">Build</option><option value="escalate">Escalate</option><option value="payoff">Payoff</option><option value="cta">CTA</option></select></label><label>Duration<input type="number" min="1" max="20" step="0.5" value={shot.duration_seconds} onChange={(event) => updateShot(index, 'duration_seconds', Number(event.target.value))} /></label></div>
          <label>LTX prompt<textarea rows={5} value={shot.prompt} onChange={(event) => updateShot(index, 'prompt', event.target.value)} placeholder="Describe subject identity, environment, action over time, camera movement, lighting, and the final composition…" /></label>
          <div className="form-grid two"><label>Negative prompt<input value={shot.negative_prompt} onChange={(event) => updateShot(index, 'negative_prompt', event.target.value)} placeholder="flicker, morphing, text…" /></label><label>Camera direction<input value={shot.camera} onChange={(event) => updateShot(index, 'camera', event.target.value)} placeholder="Low dolly push, locked subject…" /></label></div>
        </div>)}
      </div>
      <div className="manual-footer"><button className="button ghost" onClick={addShot}>+ Add shot</button><span className={Math.abs(total - project.brief.duration_seconds) < .05 ? 'duration-ok' : 'duration-bad'}>{total}s authored / {project.brief.duration_seconds}s target</span><button className="button primary large" disabled={Boolean(busy) || shots.some((shot) => !shot.prompt.trim()) || Math.abs(total - project.brief.duration_seconds) >= .05} onClick={() => act('manual-plan', () => request(`/projects/${project.id}/manual-plan`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ shots }) }), 'Exact manual storyboard saved')}>{busy === 'manual-plan' ? 'Saving…' : 'Lock exact prompts'}</button></div>
    </>}
  </section>;
}

function ShotCard({ shot, index, visualAssets, busy, act }: { shot: Json; index: number; visualAssets: Json[]; busy: string; act: (label: string, operation: () => Promise<any>, success?: string) => Promise<any> }) {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({ title: shot.data?.title || `Shot ${index + 1}`, purpose: shot.data?.purpose || 'build', duration_seconds: shot.duration_seconds, prompt: shot.prompt, negative_prompt: shot.negative_prompt || '', camera: shot.data?.camera || '', audio_mode: shot.data?.audio_mode || (shot.data?.dialogue ? 'native-dialogue' : 'ambient'), audio: shot.data?.audio || '', dialogue: shot.data?.dialogue || '', speaker: shot.data?.speaker || '', language: shot.data?.language || 'English', accent: shot.data?.accent || '', voiceover_text: shot.data?.voiceover_text || '', reference_asset_id: shot.data?.reference_asset_id || '', reference_role: shot.data?.reference_role || 'subject' });
  const locked = Boolean(shot.candidates?.length);
  const attachedReference = visualAssets.find((asset: Json) => asset.id === form.reference_asset_id);
  const dialogueWords = String(form.dialogue || '').trim().split(/\s+/).filter(Boolean).length;
  const dialogueBudget = Math.max(3, Math.floor(Number(form.duration_seconds) * 2.5));

  async function save() {
    await act(`shot-${shot.id}`, () => request(`/shots/${shot.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) }), 'Shot updated');
    setEditing(false);
  }

  return <article className="shot-card">
    <div className="shot-index"><span>{String(index + 1).padStart(2, '0')}</span><i /></div>
    <div className="shot-body">
      {shot.data?.scene_heading && <div className="scene-heading"><span>Scene {shot.data.scene_number || index + 1}</span><strong>{shot.data.scene_heading}</strong></div>}
      <div className="shot-head"><div><span className="shot-purpose">{form.purpose}</span><h3>{form.title}</h3></div><div className="shot-meta"><span>{form.duration_seconds}s</span><Pill status={shot.status} />{!locked && <button className="text-button" onClick={() => setEditing(!editing)}>{editing ? 'Close' : 'Edit'}</button>}</div></div>
      {editing ? <div className="shot-editor"><div className="form-grid three"><label>Title<input value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} /></label><label>Purpose<select value={form.purpose} onChange={(event) => setForm({ ...form, purpose: event.target.value })}><option value="hook">Hook</option><option value="build">Build</option><option value="escalate">Escalate</option><option value="payoff">Payoff</option><option value="cta">CTA</option></select></label><label>Seconds<input type="number" min="1" max="20" step=".5" value={form.duration_seconds} onChange={(event) => setForm({ ...form, duration_seconds: Number(event.target.value) })} /></label></div><label>Visual generation prompt<textarea rows={5} value={form.prompt} onChange={(event) => setForm({ ...form, prompt: event.target.value })} /></label><div className="form-grid two"><label>Negative<input value={form.negative_prompt} onChange={(event) => setForm({ ...form, negative_prompt: event.target.value })} /></label><label>Camera<input value={form.camera} onChange={(event) => setForm({ ...form, camera: event.target.value })} /></label></div><div className="shot-audio-editor"><label>Shot audio intent<select value={form.audio_mode} onChange={(event) => setForm({ ...form, audio_mode: event.target.value })}><option value="ambient">Ambience + Foley</option><option value="native-dialogue">Native LTX dialogue</option><option value="silent">Silent</option></select></label><label>Ambience / Foley<input value={form.audio} onChange={(event) => setForm({ ...form, audio: event.target.value })} placeholder="Room tone, footsteps, restrained impact…" /></label>{form.audio_mode === 'native-dialogue' && <><label>One visible speaker<input value={form.speaker} onChange={(event) => setForm({ ...form, speaker: event.target.value })} placeholder="The reporter" /></label><label>Language<input value={form.language} onChange={(event) => setForm({ ...form, language: event.target.value })} /></label><label>Accent <span className="optional-label">optional</span><input value={form.accent} onChange={(event) => setForm({ ...form, accent: event.target.value })} placeholder="Canadian English" /></label><label className="dialogue-field">Exact spoken words<textarea rows={3} value={form.dialogue} onChange={(event) => setForm({ ...form, dialogue: event.target.value })} /><span className={dialogueWords > dialogueBudget ? 'audio-budget over' : 'audio-budget'}>{dialogueWords}/{dialogueBudget} reliable words for {form.duration_seconds}s</span></label></>}<label className="voiceover-copy">Voiceover copy <span className="optional-label">mixed during export, not spoken by LTX</span><textarea rows={3} value={form.voiceover_text} onChange={(event) => setForm({ ...form, voiceover_text: event.target.value })} /></label></div><div className="shot-reference-editor"><label>Shot-specific visual reference<select value={form.reference_asset_id} onChange={(event) => setForm({ ...form, reference_asset_id: event.target.value })}><option value="">Use draft-level anchor / continuity</option>{visualAssets.map((asset: Json) => <option key={asset.id} value={asset.id}>{asset.metadata?.label || asset.filename}</option>)}</select></label><label>Reference belongs to<select value={form.reference_role} disabled={!form.reference_asset_id} onChange={(event) => setForm({ ...form, reference_role: event.target.value })}><option value="subject">Main subject</option><option value="product">Product</option><option value="shirt">Shirt</option><option value="hat">Hat</option><option value="sign">Sign / packaging</option><option value="wardrobe">Wardrobe</option><option value="location">Location</option><option value="custom">Custom</option></select></label><p>This visible reference replaces incoming clip continuity for this shot. Mention its placement in the prompt above.</p></div><button className="button primary" disabled={busy === `shot-${shot.id}` || (form.audio_mode === 'native-dialogue' && (!form.dialogue.trim() || dialogueWords > dialogueBudget))} onClick={save}>Save shot</button></div> : <><p className="shot-prompt">{shot.prompt}</p>{shot.data?.camera && <p className="shot-detail"><b>Camera</b> {shot.data.camera}</p>}<p className={`shot-audio-summary ${form.audio_mode}`}><b>{form.audio_mode === 'native-dialogue' ? 'Native dialogue' : form.audio_mode === 'silent' ? 'Silent' : 'Ambience only'}</b>{form.audio_mode === 'native-dialogue' ? `“${form.dialogue}”` : (form.audio || 'Natural ambience and Foley')}</p>{attachedReference && <div className="shot-reference-chip"><img src={`${API}/assets/${attachedReference.id}`} alt="" /><span><b>{form.reference_role}</b>{attachedReference.metadata?.label || attachedReference.filename}</span></div>}</>}
      {!editing && shot.data?.script_excerpt && <details className="script-excerpt"><summary>Original script beat</summary><p>{shot.data.script_excerpt}</p></details>}
      {locked && <p className="lock-note">Prompt locked because candidates exist. Create a new plan to change generation text.</p>}
    </div>
  </article>;
}

function CandidateRow({ shot, index, scoringEnabled, upscaleTarget, upscaleJobs, transformJobs, transformModes, inOutpaintingReady, lipdubReady, maskAssets, executionInfo, busy, act }: { shot: Json; index: number; scoringEnabled: boolean; upscaleTarget: string; upscaleJobs: Json[]; transformJobs: Json[]; transformModes: Json[]; inOutpaintingReady: boolean; lipdubReady: boolean; maskAssets: Json[]; executionInfo: Json | null; busy: string; act: (label: string, operation: () => Promise<any>, success?: string) => Promise<any> }) {
  return <div className="candidate-row"><div className="candidate-row-head"><div><span>Shot {index + 1}</span><strong>{shot.data?.title || `Shot ${index + 1}`}</strong></div><Pill status={shot.status} /></div><div className="candidate-grid">
    {(shot.candidates || []).map((candidate: Json, candidateIndex: number) => {
      const selected = shot.selected_candidate_id === candidate.id;
      const judgeUnavailable = candidate.score?.available === false || candidate.score?.judge === 'technical-fallback-v1';
      const manualReview = candidate.score?.judge === 'manual-review';
      const score = candidate.total_score == null || judgeUnavailable ? null : Math.round(candidate.total_score);
      return <article className={`candidate-card ${selected ? 'selected' : ''}`} key={candidate.id}>
        <div className="candidate-media">{candidate.artifact ? <video controls preload="metadata" src={`${API}/candidates/${candidate.id}/media`} /> : <div className="candidate-placeholder"><span>{candidate.status === 'failed' ? '!' : '◌'}</span><p>{prettyStatus(candidate.status)}</p></div>}<span className="take-label">{candidate.draft ? `Draft ${candidateIndex + 1}` : 'GGUF final'}</span>{selected && <span className="keeper-label">Keeper</span>}</div>
        <div className="candidate-info"><div><strong>{candidate.settings?.take_role?.replace(/-/g, ' ') || (candidate.draft ? 'seed variation' : 'production pass')}</strong><span>Seed {candidate.seed}</span></div>{judgeUnavailable ? <div className={`score-unavailable ${manualReview ? 'manual' : ''}`} title={candidate.score?.issues?.join('\n')}>{manualReview ? 'Manual review' : 'Judge offline'}</div> : score !== null && <div className={`score ${score >= 75 ? 'high' : score >= 60 ? 'mid' : 'low'}`}>{score}</div>}</div>
        {candidate.error && <p className="candidate-error">{candidate.error}</p>}
        {judgeUnavailable && <p className="candidate-score-note">{manualReview ? 'Scoring is intentionally off. Watch the clip and choose the keeper directly.' : 'The vision judge was unavailable; no fake score was assigned.'}</p>}
        <div className="candidate-actions"><button className={selected ? 'button selected-button' : 'button secondary'} disabled={!candidate.artifact || Boolean(busy)} onClick={() => act(`select-${candidate.id}`, () => request(`/shots/${shot.id}/select`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ candidate_id: candidate.id }) }), 'Keeper selected')}>{selected ? '✓ Selected' : 'Select keeper'}</button>{candidate.artifact && <CandidateUpscale candidateId={candidate.id} targetId={upscaleTarget} executionInfo={executionInfo} existingJob={upscaleJobs.find((job: Json) => job.candidate_id === candidate.id)} />}{candidate.artifact && <CandidateTransform candidateId={candidate.id} targetId={upscaleTarget} modes={transformModes} executionInfo={executionInfo} existingJobs={transformJobs.filter((job: Json) => job.candidate_id === candidate.id)} />}{candidate.artifact && inOutpaintingReady && <CandidateInOutpaint candidateId={candidate.id} targetId={upscaleTarget} executionInfo={executionInfo} existingJobs={transformJobs.filter((job: Json) => job.candidate_id === candidate.id && job.payload?.mode === 'in-outpainting')} maskAssets={maskAssets} />}{candidate.artifact && lipdubReady && <CandidateLipDub candidateId={candidate.id} duration={Number(candidate.settings?.duration_seconds || shot.duration_seconds || 5)} targetId={upscaleTarget} executionInfo={executionInfo} existingJobs={transformJobs.filter((job: Json) => job.candidate_id === candidate.id && job.payload?.mode === 'lipdub')} />}{scoringEnabled && candidate.artifact && (candidate.status !== 'scored' || judgeUnavailable) && <button className="text-button" disabled={Boolean(busy)} onClick={() => act(`rescore-${candidate.id}`, () => request(`/candidates/${candidate.id}/rescore`, { method: 'POST' }), 'Scoring queued')}>Rescore</button>}</div>
      </article>;
    })}
  </div></div>;
}

function CandidateInOutpaint({ candidateId, targetId, executionInfo, existingJobs, maskAssets }: { candidateId: string; targetId: string; executionInfo: Json | null; existingJobs: Json[]; maskAssets: Json[] }) {
  const latestJob = existingJobs[0] || null;
  const [open, setOpen] = useState(Boolean(latestJob));
  const [operation, setOperation] = useState<'inpaint' | 'outpaint'>(latestJob?.payload?.operation === 'outpaint' ? 'outpaint' : 'inpaint');
  const [maskAssetId, setMaskAssetId] = useState(String(latestJob?.payload?.mask_asset_id || maskAssets[0]?.id || ''));
  const [prompt, setPrompt] = useState(String(latestJob?.payload?.prompt || ''));
  const [direction, setDirection] = useState(String(latestJob?.payload?.outpaint_direction || 'all'));
  const [expansion, setExpansion] = useState(Number(latestJob?.payload?.expansion_percent || 25));
  const [dilation, setDilation] = useState(Number(latestJob?.payload?.mask_dilation || 15));
  const [job, setJob] = useState<Json | null>(latestJob);
  const [error, setError] = useState('');
  const target = executionInfo?.targets?.find((item: Json) => item.id === targetId);
  const active = Boolean(job && ACTIVE_JOBS.has(job.status));

  useEffect(() => {
    if (!maskAssetId && maskAssets[0]?.id) setMaskAssetId(maskAssets[0].id);
  }, [maskAssetId, maskAssets]);
  useEffect(() => {
    if (!latestJob || (job && latestJob.id !== job.id)) return;
    setJob(latestJob);
    setOpen(true);
  }, [latestJob?.id, latestJob?.status, latestJob?.progress]);
  useEffect(() => {
    if (!job?.id || !ACTIVE_JOBS.has(job.status)) return;
    const interval = window.setInterval(() => request(`/jobs/${job.id}`).then(setJob).catch((cause) => setError(cause.message)), 2000);
    return () => window.clearInterval(interval);
  }, [job?.id, job?.status]);

  async function start() {
    setError('');
    try {
      const payload: Json = {
        mode: 'in-outpainting', candidate_id: candidateId, target_id: targetId,
        operation, strength: 1, prompt: prompt.trim(),
        ...(operation === 'inpaint'
          ? { mask_asset_id: maskAssetId, mask_dilation: dilation }
          : { outpaint_direction: direction, expansion_percent: expansion }),
      };
      const value = await request('/creative-lab/transforms', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      setJob(value.job);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'In/Outpainting could not start'); }
  }

  async function retry() {
    if (!job?.id) return;
    setError('');
    try {
      await request(`/jobs/${job.id}/retry`, { method: 'POST' });
      setJob(await request(`/jobs/${job.id}`));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'In/Outpainting retry failed'); }
  }

  if (!open) return <button className="text-button inoutpaint-open" onClick={() => setOpen(true)}>In / Outpaint</button>;
  return <div className="inoutpaint-tool">
    <div className="inoutpaint-head"><strong>In / Outpainting</strong><span>Official two-stage masked edit · source audio preserved</span><button className="text-button" disabled={active} onClick={() => setOpen(false)}>Close</button></div>
    <div className="inoutpaint-kind"><button className={operation === 'inpaint' ? 'active' : ''} disabled={active} onClick={() => setOperation('inpaint')}><strong>Inpaint</strong><span>Replace a masked region</span></button><button className={operation === 'outpaint' ? 'active' : ''} disabled={active} onClick={() => setOperation('outpaint')}><strong>Outpaint</strong><span>Extend beyond the frame</span></button></div>
    <label className="inoutpaint-prompt">Describe only the new region<input value={prompt} disabled={active} onChange={(event) => setPrompt(event.target.value)} placeholder="A clean brick wall continues behind the subject…" /></label>
    {operation === 'inpaint' ? <>
      <label>Mask asset<select value={maskAssetId} disabled={active} onChange={(event) => setMaskAssetId(event.target.value)}><option value="">Upload a mask in Production assets…</option>{maskAssets.map((asset: Json) => <option value={asset.id} key={asset.id}>{asset.metadata?.label || asset.filename}</option>)}</select></label>
      <label>Edge context · {dilation}px<input type="range" min="0" max="15" step="1" value={dilation} disabled={active} onChange={(event) => setDilation(Number(event.target.value))} /></label>
      <small>White pixels are regenerated; black pixels are preserved. The mask must match the source dimensions. Include shadows, reflections, and contact edges.</small>
    </> : <>
      <label>Extend<select value={direction} disabled={active} onChange={(event) => setDirection(event.target.value)}><option value="all">All sides</option><option value="left">Left</option><option value="right">Right</option><option value="top">Top</option><option value="bottom">Bottom</option></select></label>
      <label>Expansion · {expansion}%<input type="range" min="10" max="100" step="5" value={expansion} disabled={active} onChange={(event) => setExpansion(Number(event.target.value))} /></label>
      <small>The source is padded locally, then the new canvas is generated with the official boundary-refinement pass.</small>
    </>}
    {job?.status === 'succeeded' ? <div className="transform-result inoutpaint-result"><video controls loop preload="metadata" src={`${API}/jobs/${job.id}/output`} /><div><a className="text-button transform-download" href={`${API}/jobs/${job.id}/output`}>Download edited clip</a><button className="text-button creative-transform-button" disabled={!target?.available} onClick={() => { setJob(null); setError(''); }}>Make another edit</button></div></div> : <button className="button secondary inoutpaint-start" disabled={!target?.available || active || (operation === 'inpaint' && !maskAssetId)} title={target?.unavailable_reason || 'Run the isolated two-stage masked workflow'} onClick={start}>{active ? `Rendering ${operation} · ${Math.round(Number(job?.progress || 0) * 100)}%` : `Start ${operation}`}</button>}
    {job?.status === 'failed' && <button className="text-button retry-upscale" disabled={!target?.available} onClick={retry}>Retry masked edit</button>}
    {(error || job?.status === 'failed') && <span className="inline-job-error" title={error || job?.error}>{error || job?.error || 'In/Outpainting failed'}</span>}
  </div>;
}

function CandidateLipDub({ candidateId, duration, targetId, executionInfo, existingJobs }: { candidateId: string; duration: number; targetId: string; executionInfo: Json | null; existingJobs: Json[] }) {
  const latestJob = existingJobs[0] || null;
  const [open, setOpen] = useState(Boolean(latestJob));
  const [language, setLanguage] = useState(String(latestJob?.payload?.language || 'English'));
  const [dialogue, setDialogue] = useState(String(latestJob?.payload?.dialogue || ''));
  const [scenePrompt, setScenePrompt] = useState(String(latestJob?.payload?.prompt || ''));
  const [job, setJob] = useState<Json | null>(latestJob);
  const [error, setError] = useState('');
  const target = executionInfo?.targets?.find((item: Json) => item.id === targetId);
  const active = Boolean(job && ACTIVE_JOBS.has(job.status));
  const wordCount = dialogue.trim().split(/\s+/).filter(Boolean).length;
  const reliableWords = Math.max(3, Math.floor(Math.min(duration, 5.04) * 2.5));

  useEffect(() => {
    if (!latestJob || (job && latestJob.id !== job.id)) return;
    setJob(latestJob);
    setOpen(true);
  }, [latestJob?.id, latestJob?.status, latestJob?.progress]);
  useEffect(() => {
    if (!job?.id || !ACTIVE_JOBS.has(job.status)) return;
    const interval = window.setInterval(() => request(`/jobs/${job.id}`).then(setJob).catch((cause) => setError(cause.message)), 2000);
    return () => window.clearInterval(interval);
  }, [job?.id, job?.status]);

  async function start() {
    setError('');
    try {
      const value = await request('/creative-lab/transforms', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'lipdub', candidate_id: candidateId, target_id: targetId,
          strength: 1, language: language.trim(), dialogue: dialogue.trim(),
          prompt: scenePrompt.trim(),
        }),
      });
      setJob(value.job);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'LipDub could not start'); }
  }

  async function retry() {
    if (!job?.id) return;
    setError('');
    try {
      await request(`/jobs/${job.id}/retry`, { method: 'POST' });
      setJob(await request(`/jobs/${job.id}`));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'LipDub retry failed'); }
  }

  if (!open) return <button className="text-button lipdub-open" onClick={() => setOpen(true)}>LipDub</button>;
  return <div className="lipdub-tool">
    <div className="lipdub-head"><strong>LipDub speech replacement</strong><span>One speaker · regenerated voice + lip motion</span><button className="text-button" disabled={active} onClick={() => setOpen(false)}>Close</button></div>
    <label>Spoken language<input value={language} disabled={active} onChange={(event) => setLanguage(event.target.value)} placeholder="English, French, Japanese…" /></label>
    <label className="lipdub-scene">Visual context <span className="optional-label">optional</span><input value={scenePrompt} disabled={active} onChange={(event) => setScenePrompt(event.target.value)} placeholder="A determined woman in a racing suit speaks inside a car…" /></label>
    <label className="lipdub-dialogue">Exact desired words<textarea rows={3} value={dialogue} disabled={active} onChange={(event) => setDialogue(event.target.value)} placeholder="Enter only the final line in the target language's native script…" /><span className={wordCount > reliableWords ? 'audio-budget over' : 'audio-budget'}>{wordCount}/{reliableWords} conservative words for this clip</span></label>
    <small>The source must already contain one visible speaker and their voice. Keep the replacement close to the original timing; LipDub does not translate for you.</small>
    {job?.status === 'succeeded' ? <div className="transform-result lipdub-result"><video controls loop preload="metadata" src={`${API}/jobs/${job.id}/output`} /><div><a className="text-button transform-download" href={`${API}/jobs/${job.id}/output`}>Download dubbed clip</a><button className="text-button creative-transform-button" disabled={!target?.available} onClick={() => { setJob(null); setError(''); }}>Try another line</button></div></div> : <button className="button secondary lipdub-start" disabled={!target?.available || active || !dialogue.trim() || !language.trim()} title={target?.unavailable_reason || 'Run two-stage LipDub with source voice identity tokens'} onClick={start}>{active ? `Rendering LipDub · ${Math.round(Number(job?.progress || 0) * 100)}%` : 'Generate LipDub'}</button>}
    {job?.status === 'failed' && <button className="text-button retry-upscale" disabled={!target?.available} onClick={retry}>Retry LipDub</button>}
    {(error || job?.status === 'failed') && <span className="inline-job-error" title={error || job?.error}>{error || job?.error || 'LipDub failed'}</span>}
  </div>;
}

function CandidateTransform({ candidateId, targetId, modes, executionInfo, existingJobs }: { candidateId: string; targetId: string; modes: Json[]; executionInfo: Json | null; existingJobs: Json[] }) {
  const availableModes = modes.map((mode: Json) => mode.id as TransformMode).filter((mode: TransformMode) => TRANSFORM_UI[mode]);
  const [mode, setMode] = useState<TransformMode>(availableModes.includes('day-to-night') ? 'day-to-night' : (availableModes[0] || 'day-to-night'));
  const [prompt, setPrompt] = useState('');
  const matchingJob = existingJobs.find((item: Json) => item.payload?.mode === mode);
  const [job, setJob] = useState<Json | null>(matchingJob || null);
  const matchingFoleyJob = job?.id ? existingJobs.find((item: Json) => item.payload?.mode === 'foley-v2a' && item.payload?.source_job_id === job.id) : null;
  const [processedFoleyJob, setProcessedFoleyJob] = useState<Json | null>(matchingFoleyJob || null);
  const [error, setError] = useState('');
  const target = executionInfo?.targets?.find((item: Json) => item.id === targetId);
  useEffect(() => {
    setJob(matchingJob || null);
    setError('');
  }, [mode, matchingJob?.id, matchingJob?.status, matchingJob?.progress]);
  useEffect(() => {
    if (!job?.id || !ACTIVE_JOBS.has(job.status)) return;
    const interval = window.setInterval(() => request(`/jobs/${job.id}`).then(setJob).catch((cause) => setError(cause.message)), 2000);
    return () => window.clearInterval(interval);
  }, [job?.id, job?.status]);
  useEffect(() => {
    if (matchingFoleyJob && (!processedFoleyJob || matchingFoleyJob.id === processedFoleyJob.id)) setProcessedFoleyJob(matchingFoleyJob);
  }, [matchingFoleyJob?.id, matchingFoleyJob?.status, matchingFoleyJob?.progress]);
  useEffect(() => {
    if (!processedFoleyJob?.id || !ACTIVE_JOBS.has(processedFoleyJob.status)) return;
    const interval = window.setInterval(() => request(`/jobs/${processedFoleyJob.id}`).then(setProcessedFoleyJob).catch((cause) => setError(cause.message)), 2000);
    return () => window.clearInterval(interval);
  }, [processedFoleyJob?.id, processedFoleyJob?.status]);
  async function start() {
    setError('');
    try {
      const value = await request('/creative-lab/transforms', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode, candidate_id: candidateId, target_id: targetId, strength: ui.strength || 1, prompt: prompt.trim() }) });
      setJob(value.job);
    } catch (cause) { setError(cause instanceof Error ? cause.message : `${TRANSFORM_UI[mode].label} could not start`); }
  }
  async function retry() {
    if (!job?.id) return;
    setError('');
    try {
      await request(`/jobs/${job.id}/retry`, { method: 'POST' });
      setJob(await request(`/jobs/${job.id}`));
    } catch (cause) { setError(cause instanceof Error ? cause.message : `${TRANSFORM_UI[mode].label} retry failed`); }
  }
  async function addProcessedFoley() {
    if (!job?.id) return;
    setError('');
    try {
      const value = await request('/creative-lab/transforms', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: 'foley-v2a', source_job_id: job.id, target_id: targetId, strength: 1, prompt: prompt.trim() }) });
      setProcessedFoleyJob(value.job);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Matched water sound could not start'); }
  }
  if (!availableModes.length) return null;
  const active = Boolean(job && ACTIVE_JOBS.has(job.status));
  const ui = TRANSFORM_UI[mode];
  return <div className="creative-transform-picker"><select value={mode} disabled={active} onChange={(event) => { setMode(event.target.value as TransformMode); setPrompt(''); setProcessedFoleyJob(null); }}>{availableModes.map((value: TransformMode) => <option value={value} key={value}>{TRANSFORM_UI[value].label}</option>)}</select><input value={prompt} disabled={active} onChange={(event) => setPrompt(event.target.value)} placeholder={ui.placeholder} />{job?.status === 'succeeded' ? <div className="transform-result"><video controls loop preload="metadata" src={`${API}/jobs/${job.id}/output`} /><div><a className="text-button transform-download" href={`${API}/jobs/${job.id}/output`}>Download {ui.label}</a>{mode === 'foley-v2a' && <button className="text-button creative-transform-button" disabled={!target?.available} title="Foley is seed-sensitive; render another take and choose by synchronization, not loudness" onClick={start}>Try another seed</button>}{mode === 'water-simulation' && (processedFoleyJob?.status === 'succeeded' ? <a className="text-button transform-download" href={`${API}/jobs/${processedFoleyJob.id}/output`}>Download with matched Foley</a> : <button className="text-button creative-transform-button" disabled={!target?.available || Boolean(processedFoleyJob && ACTIVE_JOBS.has(processedFoleyJob.status))} onClick={addProcessedFoley}>{processedFoleyJob && ACTIVE_JOBS.has(processedFoleyJob.status) ? `Adding water sound · ${Math.round(Number(processedFoleyJob.progress || 0) * 100)}%` : 'Add matched water sound'}</button>)}</div>{processedFoleyJob?.status === 'failed' && <span className="inline-job-error">{processedFoleyJob.error || 'Matched water sound failed'}</span>}</div> : <button className="text-button creative-transform-button" disabled={!target?.available || active || Boolean(ui.promptRequired && !prompt.trim())} title={`Apply the installed LTX 2.3 ${ui.label} as an isolated pass`} onClick={start}>{active ? `${ui.label} · ${Math.round(Number(job?.progress || 0) * 100)}%` : ui.action}</button>}{job?.status === 'failed' && <button className="text-button retry-upscale" disabled={!target?.available} onClick={retry}>Retry {ui.label}</button>}{(error || job?.status === 'failed') && <span className="inline-job-error" title={error || job?.error}>{error || job?.error || `${ui.label} failed`}</span>}</div>;
}

function CandidateUpscale({ candidateId, targetId, executionInfo, existingJob }: { candidateId: string; targetId: string; executionInfo: Json | null; existingJob?: Json }) {
  const [job, setJob] = useState<Json | null>(existingJob || null);
  const [error, setError] = useState('');
  const target = executionInfo?.targets?.find((item: Json) => item.id === targetId);
  useEffect(() => {
    if (existingJob && (!job || existingJob.id === job.id)) setJob(existingJob);
  }, [existingJob?.id, existingJob?.status, existingJob?.progress]);
  useEffect(() => {
    if (!job?.id || !ACTIVE_JOBS.has(job.status)) return;
    const interval = window.setInterval(() => request(`/jobs/${job.id}`).then(setJob).catch((cause) => setError(cause.message)), 2000);
    return () => window.clearInterval(interval);
  }, [job?.id, job?.status]);
  async function start(scale: 2 | 4) {
    setError('');
    try {
      const value = await request('/upscales', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ candidate_id: candidateId, target_id: targetId, scale }) });
      setJob(value.job);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Upscale could not start'); }
  }
  async function recover() {
    if (!job?.id) return;
    setError('');
    try {
      setJob(await request(`/jobs/${job.id}/recover-upscale`, { method: 'POST' }));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Completed upscale could not be recovered'); }
  }
  if (job?.status === 'succeeded') return <a className="text-button upscale-download" href={`${API}/jobs/${job.id}/output`}>Download {job.payload?.scale || 2}×</a>;
  const active = Boolean(job && ACTIVE_JOBS.has(job.status));
  return <><button className="text-button" disabled={!target?.available || active} title={target?.unavailable_reason || `Run x2 on ${target?.label || targetId}`} onClick={() => start(2)}>{active ? `Pixel ${job?.payload?.scale || 2}× · ${Math.round(Number(job?.progress || 0) * 100)}%` : 'Pixel 2×'}</button><button className="text-button" disabled={!target?.available || active} title="More generative x4 Pixel Spatial mode; delivery is capped for 24 GB VRAM" onClick={() => start(4)}>Pixel 4×</button>{job?.status === 'failed' && <button className="text-button retry-upscale" disabled={!target?.available} onClick={recover}>Recover completed Pixel output</button>}{(error || job?.status === 'failed') && <span className="inline-job-error" title={error || job?.error}>{error || job?.error || 'Upscale failed'}</span>}</>;
}

function FinishPanel({ project, shots, selectedDrafts, selectedFinals, execution, executionInfo, busy, act }: { project: Json; shots: Json[]; selectedDrafts: number; selectedFinals: number; execution: ExecutionPreferences; executionInfo: Json | null; busy: string; act: (label: string, operation: () => Promise<any>, success?: string) => Promise<any> }) {
  const [captions, setCaptions] = useState('');
  const [voiceoverAsset, setVoiceoverAsset] = useState('');
  const [originalAudioVolume, setOriginalAudioVolume] = useState(.35);
  const [logoAsset, setLogoAsset] = useState('');
  const [logoPosition, setLogoPosition] = useState('bottom-right');
  const [logoSize, setLogoSize] = useState(14);
  const [logoOpacity, setLogoOpacity] = useState(1);
  const allDrafts = Boolean(shots.length && selectedDrafts === shots.length);
  const allFinals = Boolean(shots.length && selectedFinals === shots.length);
  const readyExport = allFinals;
  const logoAssets = (project.assets || []).filter((asset: Json) => String(asset.mime_type || '').startsWith('image/') && (asset.metadata?.brand_role === 'logo' || (asset.kind === 'brand' && !asset.metadata?.brand_role)));
  const voiceoverAssets = (project.assets || []).filter((asset: Json) => asset.kind === 'voiceover');
  return <section className="panel finish-panel">
    <div className="section-head"><div><span className="eyebrow">Final pass</span><h2>Finish on the stable GGUF workflow</h2><p>Selected drafts are rerendered through the 4-step generation + 3-step upscale path. FP8 stays off.</p></div></div>
    <div className="finish-grid">
      <div className={allDrafts ? 'finish-card ready' : 'finish-card'}><span className="finish-number">01</span><h3>Render winners</h3><p>{allDrafts ? `${selectedFinals}/${shots.length} final clips ready · ${executionLabel(executionInfo, execution.generation)}.` : `Select a draft for all ${shots.length} shots first.`}</p><button className="button primary" disabled={!allDrafts || allFinals || Boolean(busy)} onClick={() => act('finals', () => request(`/projects/${project.id}/render-winners`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ execution_target: execution.generation }) }), 'GGUF final renders queued')}>{busy === 'finals' ? 'Checking target…' : allFinals ? 'Finals complete' : 'Render GGUF finals'}</button></div>
      <div className={readyExport ? 'finish-card ready' : 'finish-card'}><span className="finish-number">02</span><h3>Assemble export</h3><label>Optional captions<textarea rows={3} value={captions} onChange={(event) => setCaptions(event.target.value)} placeholder="One caption per line…" /></label><div className="voiceover-controls"><label>Exact voiceover track<select value={voiceoverAsset} onChange={(event) => setVoiceoverAsset(event.target.value)}><option value="">None</option>{voiceoverAssets.map((asset: Json) => <option key={asset.id} value={asset.id}>{asset.filename}</option>)}</select></label>{voiceoverAsset && <label>LTX ambience under voice · {Math.round(originalAudioVolume * 100)}%<input type="range" min="0" max="1" step=".05" value={originalAudioVolume} onChange={(event) => setOriginalAudioVolume(Number(event.target.value))} /></label>}</div>{voiceoverAsset && <p className="voiceover-note">The uploaded recording is mixed locally as the authoritative spoken track. LTX supplies only the background ambience.</p>}<div className="exact-logo-controls"><label>Exact logo overlay<select value={logoAsset} onChange={(event) => setLogoAsset(event.target.value)}><option value="">None</option>{logoAssets.map((asset: Json) => <option key={asset.id} value={asset.id}>{asset.metadata?.label || asset.filename}</option>)}</select></label>{logoAsset && <><label>Position<select value={logoPosition} onChange={(event) => setLogoPosition(event.target.value)}><option value="top-left">Top left</option><option value="top-right">Top right</option><option value="bottom-left">Bottom left</option><option value="bottom-right">Bottom right</option></select></label><label>Width · {logoSize}%<input type="range" min="3" max="40" value={logoSize} onChange={(event) => setLogoSize(Number(event.target.value))} /></label><label>Opacity · {Math.round(logoOpacity * 100)}%<input type="range" min="0.1" max="1" step="0.05" value={logoOpacity} onChange={(event) => setLogoOpacity(Number(event.target.value))} /></label></>}</div>{logoAsset && <p className="exact-logo-note">Pixel-exact corner overlay. Logos generated on moving shirts, hats, or signs remain part of the LTX reference pass.</p>}<button className="button primary" disabled={!readyExport || Boolean(busy)} onClick={() => act('export', () => request(`/projects/${project.id}/exports`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ platform: project.brief.platform, aspect_ratio: project.brief.aspect_ratio, captions, burn_captions: Boolean(captions), voiceover_asset_id: voiceoverAsset || null, original_audio_volume: voiceoverAsset ? originalAudioVolume : .8, voiceover_volume: 1, logo_asset_id: logoAsset || null, logo_position: logoPosition, logo_width_percent: logoSize, logo_opacity: logoOpacity }) }), 'Export queued')}>{busy === 'export' ? 'Assembling…' : 'Build final video'}</button></div>
    </div>
    {project.exports?.length ? <div className="exports"><h3>Exports</h3>{project.exports.map((item: Json) => <div key={item.id}><div><Pill status={item.status} /><span>{item.platform} · {relativeTime(item.created_at)}</span></div>{item.status === 'ready' && <a className="button success" href={`${API}/exports/${item.id}/download`}>Download MP4</a>}</div>)}</div> : null}
  </section>;
}

function JobDrawer({ jobs, busy, act }: { jobs: Json[]; busy: string; act: (label: string, operation: () => Promise<any>, success?: string) => Promise<any> }) {
  return <section className="job-drawer"><div className="section-head"><div><span className="eyebrow">Durable queue</span><h2>Recent jobs</h2></div></div><div className="job-list">{jobs.slice(0, 30).map((job) => <div className="job-row" key={job.id}><div className="job-icon">{job.kind.slice(0, 1).toUpperCase()}</div><div><strong>{prettyStatus(job.kind)}</strong><span>{job.payload?.execution_target || job.payload?.target_id || job.worker_id || job.lane} · attempt {job.attempts}/{job.max_attempts}</span>{job.error && <em>{job.error}</em>}</div><Pill status={job.status} /><div className="job-actions">{ACTIVE_JOBS.has(job.status) && <button className="text-button" disabled={Boolean(busy)} onClick={() => act(`cancel-${job.id}`, () => request(`/jobs/${job.id}/cancel`, { method: 'POST' }), 'Cancellation requested')}>Cancel</button>}{['failed', 'cancelled'].includes(job.status) && <button className="text-button" disabled={Boolean(busy)} onClick={() => act(`retry-${job.id}`, () => request(`/jobs/${job.id}/retry`, { method: 'POST' }), 'Job queued again')}>Retry</button>}</div></div>)}</div></section>;
}

function QuickFinish({ chain, resolution, onChain, onError }: { chain: Json; resolution: Json; onChain: (value: Json) => void; onError: (message: string) => void }) {
  const [open, setOpen] = useState(Boolean(chain.finished_path));
  const [delivery, setDelivery] = useState('source');
  const [captions, setCaptions] = useState('');
  const [transition, setTransition] = useState(.15);
  const [sourceAudio, setSourceAudio] = useState(0);
  const [music, setMusic] = useState<File | null>(null);
  const [voiceover, setVoiceover] = useState<File | null>(null);
  const [logo, setLogo] = useState<File | null>(null);
  const [logoPosition, setLogoPosition] = useState('bottom-right');
  const [logoSize, setLogoSize] = useState(14);
  const [logoOpacity, setLogoOpacity] = useState(1);
  const [finishing, setFinishing] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const kept = (chain.clips || []).filter((clip: Json) => clip.status === 'done');
  const finishJob = chain.finish_job;
  const remoteFinishing = Boolean(finishJob && ACTIVE_JOBS.has(finishJob.status));
  const isFinishing = finishing || remoteFinishing;
  const finishError = chain.finish_metadata?.error || (finishJob?.status === 'failed' ? String(finishJob.error || 'Social finish failed').split('\n')[0] : '');

  useEffect(() => {
    if (!isFinishing) return;
    const timer = window.setInterval(() => setElapsed((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [isFinishing]);

  useEffect(() => {
    if (!finishJob?.id || !remoteFinishing) return;
    setOpen(true);
    const started = new Date(finishJob.started_at || finishJob.created_at || Date.now()).getTime();
    setElapsed(Math.max(0, Math.floor((Date.now() - started) / 1000)));
    const poll = window.setInterval(() => {
      request(`/chain/${chain.id}`).then(onChain).catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(poll);
  }, [chain.id, finishJob?.id, remoteFinishing, onChain]);

  async function finish() {
    if (!kept.length || isFinishing) return;
    const data = new FormData();
    if (delivery === 'source') {
      data.append('platform', 'custom');
      data.append('width', String(resolution.width * 2));
      data.append('height', String(resolution.height * 2));
    } else {
      data.append('platform', delivery);
    }
    data.append('captions', captions);
    data.append('transition_seconds', String(transition));
    data.append('original_audio_volume', String(sourceAudio));
    data.append('music_volume', '.18');
    data.append('voiceover_volume', '1');
    data.append('logo_position', logoPosition);
    data.append('logo_width_percent', String(logoSize));
    data.append('logo_opacity', String(logoOpacity));
    if (music) data.append('music', music);
    if (voiceover) data.append('voiceover', voiceover);
    if (logo) data.append('logo', logo);
    setFinishing(true); setElapsed(0); onError('');
    try {
      const value = await request(`/chain/${chain.id}/finish`, { method: 'POST', body: data });
      onChain(value.chain);
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : 'Social finish failed');
    } finally {
      setFinishing(false);
    }
  }

  return <section className="panel quick-finish">
    <div className="section-head"><div><span className="eyebrow">Local finishing</span><h2>Finish social cut</h2><p>Turn the {kept.length}-clip story into a post-ready MP4 without another GPU render.</p></div><button className="button secondary" onClick={() => setOpen(!open)}>{open ? 'Hide controls' : chain.finished_path ? 'Edit finish' : 'Open finisher'}</button></div>
    {chain.finished_path && <div className="finished-social"><video controls preload="metadata" src={`${API}/chain/${chain.id}/finished?v=${encodeURIComponent(chain.updated_at || '')}`} /><div><span>Social export ready</span><strong>{chain.finish_metadata?.width} × {chain.finish_metadata?.height} · {chain.finish_metadata?.duration_seconds || '—'}s</strong><small>{chain.finish_metadata?.captions_burned ? 'Captions burned · ' : ''}{chain.finish_metadata?.voiceover_mixed ? 'Voiceover mixed · ' : ''}{chain.finish_metadata?.music_mixed ? 'Music mixed · ' : ''}{chain.finish_metadata?.logo_overlaid ? 'Logo applied' : ''}</small><a className="button success" href={`${API}/chain/${chain.id}/finished`} download>Download post-ready MP4</a></div></div>}
    {remoteFinishing && <div className="finish-job-progress"><div><i /><span>Local export · {Math.round(Number(finishJob.progress || 0) * 100)}%</span></div><strong>Rendering continues safely if you switch tabs or reload · {elapsed}s</strong></div>}
    {finishError && <div className="chain-failure"><strong>Social finish failed</strong><span>{finishError}</span></div>}
    {open && <div className="quick-finish-controls">
      <div className="finish-platforms"><button className={delivery === 'source' ? 'active' : ''} onClick={() => setDelivery('source')}><strong>Match source</strong><span>{resolution.width * 2} × {resolution.height * 2}</span></button>{[['reels', 'Reels'], ['tiktok', 'TikTok'], ['shorts', 'Shorts'], ['youtube', 'YouTube'], ['x', 'X']].map(([value, label]) => <button key={value} className={delivery === value ? 'active' : ''} onClick={() => setDelivery(value)}><strong>{label}</strong><span>{['youtube', 'x'].includes(value) ? '1280 × 720' : '720 × 1280'}</span></button>)}</div>
      <label>On-screen captions <span className="optional-label">one beat per line, timed evenly</span><textarea rows={4} value={captions} onChange={(event) => setCaptions(event.target.value)} placeholder={'STOP SCROLLING\nHERE IS THE TURN\nTHE FINAL PAYOFF'} /></label>
      <div className="quick-finish-files"><label>Exact voiceover <span>WAV, MP3, M4A</span><input type="file" accept="audio/*" onChange={(event) => setVoiceover(event.target.files?.[0] || null)} /><small>{voiceover?.name || 'No voiceover'}</small></label><label>Music bed <span>WAV, MP3, M4A</span><input type="file" accept="audio/*" onChange={(event) => setMusic(event.target.files?.[0] || null)} /><small>{music?.name || 'No music'}</small></label><label>Exact logo <span>PNG recommended</span><input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => setLogo(event.target.files?.[0] || null)} /><small>{logo?.name || 'No logo'}</small></label></div>
      <div className="quick-finish-mix"><label>Clip transition · {transition.toFixed(2)}s<input type="range" min="0" max=".75" step=".05" value={transition} onChange={(event) => setTransition(Number(event.target.value))} /></label><label>Original LTX audio · {Math.round(sourceAudio * 100)}%<input type="range" min="0" max="1" step=".05" value={sourceAudio} onChange={(event) => setSourceAudio(Number(event.target.value))} /></label>{logo && <><label>Logo position<select value={logoPosition} onChange={(event) => setLogoPosition(event.target.value)}><option value="top-left">Top left</option><option value="top-right">Top right</option><option value="bottom-left">Bottom left</option><option value="bottom-right">Bottom right</option></select></label><label>Logo width · {logoSize}%<input type="range" min="3" max="40" value={logoSize} onChange={(event) => setLogoSize(Number(event.target.value))} /></label><label>Logo opacity · {Math.round(logoOpacity * 100)}%<input type="range" min=".1" max="1" step=".05" value={logoOpacity} onChange={(event) => setLogoOpacity(Number(event.target.value))} /></label></>}</div>
      <p className="quick-finish-note">Original model audio starts muted to prevent accidental speech. Uploaded voiceover and music are authoritative and mixed locally at social loudness.</p>
      <button className="button primary large" disabled={isFinishing || !kept.length} onClick={finish}>{isFinishing ? `Finishing locally · ${elapsed}s` : chain.finished_path ? 'Rebuild social cut' : finishError ? 'Retry social cut' : 'Build social cut'}</button>
    </div>}
  </section>;
}

function QuickGenerate({ onError, execution, executionInfo }: { onError: (message: string) => void; execution: ExecutionPreferences; executionInfo: Json | null }) {
  const [saved] = useState<Json | null>(() => savedQuickGeneration());
  const [savedChain] = useState<Json | null>(() => savedQuickChain());
  const [mode, setMode] = useState<'t2v' | 'i2v'>(saved?.mode || savedChain?.mode || 't2v');
  const [prompt, setPrompt] = useState(saved?.prompt || savedChain?.prompt || '');
  const [negative, setNegative] = useState(saved?.negative || savedChain?.negative || '');
  const [resolution, setResolution] = useState(
    RESOLUTIONS.find((item) => item.width === (saved?.width || savedChain?.width) && item.height === (saved?.height || savedChain?.height)) || RESOLUTIONS[1]
  );
  const [duration, setDuration] = useState(saved?.duration || savedChain?.duration || 5);
  const [audioMode, setAudioMode] = useState<AudioMode>(() => migratedQuickAudioMode(saved?.audioMode || savedChain?.audioMode));
  const [dialogue, setDialogue] = useState(saved?.dialogue || savedChain?.dialogue || '');
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [generating, setGenerating] = useState(Boolean(saved));
  const [elapsed, setElapsed] = useState(
    saved ? Math.max(0, Math.floor((Date.now() - Number(saved.startedAt)) / 1000)) : 0
  );
  const [promptId, setPromptId] = useState(saved?.promptId || '');
  const [activeGenerationTarget, setActiveGenerationTarget] = useState(saved?.executionTarget || savedChain?.executionTarget || execution.generation);
  const [result, setResult] = useState(savedChain?.remoteFilename || '');
  const [chain, setChain] = useState<Json | null>(null);
  const [restoringChain, setRestoringChain] = useState(Boolean(savedChain?.chainId && !saved));
  const [nextPrompt, setNextPrompt] = useState('');
  const [nextDialogue, setNextDialogue] = useState('');
  const [continuing, setContinuing] = useState(false);
  const [continuationSeenRemote, setContinuationSeenRemote] = useState(false);
  const [continuationWatchStartedAt, setContinuationWatchStartedAt] = useState(0);
  const [continuationElapsed, setContinuationElapsed] = useState(0);
  const [recent, setRecent] = useState<Json[] | null>(null);
  const [recentChains, setRecentChains] = useState<Json[]>([]);
  const [loadingRecent, setLoadingRecent] = useState(false);
  const [upscaleJob, setUpscaleJob] = useState<Json | null>(null);
  const [upscaleScale, setUpscaleScale] = useState<2 | 4>(2);
  const [upscaleElapsed, setUpscaleElapsed] = useState(0);
  const [compilePreview, setCompilePreview] = useState<Json | null>(null);
  const [showLtxPreview, setShowLtxPreview] = useState(false);
  const upscaleTarget = executionInfo?.targets?.find((target: Json) => target.id === execution.postUpscale);
  const generationMachine = executionLabel(executionInfo, (generating || result) ? activeGenerationTarget : execution.generation);
  const continuationMachine = executionLabel(executionInfo, execution.continuation, 'Automatic upload routing');
  const spokenWords = dialogue.trim() || quotedWords(prompt);
  const nextSpokenWords = nextDialogue.trim() || quotedWords(nextPrompt);
  const scriptShape = compilePreview?.script_shape;
  const looksLikeScript = Boolean(scriptShape?.looks_like_script);

  function applyChain(value: Json) {
    const opening = (value.clips || []).find((clip: Json) => clip.position === 0 && clip.status === 'done');
    const active = (value.clips || []).find((clip: Json) => clip.status === 'generating');
    setChain(value);
    setResult(opening?.remote_filename || '');
    if (opening?.prompt) setPrompt(opening.prompt);
    if (active) {
      setContinuing(true); setContinuationSeenRemote(true);
      const start = new Date(active.created_at || active.updated_at || Date.now()).getTime();
      setContinuationWatchStartedAt(start);
      setContinuationElapsed(Math.max(0, Math.floor((Date.now() - start) / 1000)));
    }
  }

  useEffect(() => {
    if (!savedChain?.chainId || saved) { setRestoringChain(false); return; }
    request(`/chain/${savedChain.chainId}`).then(applyChain).catch(() => {
      localStorage.removeItem(QUICK_CHAIN_KEY);
    }).finally(() => setRestoringChain(false));
  }, []);

  useEffect(() => {
    if (!chain?.id) return;
    const opening = (chain.clips || []).find((clip: Json) => clip.position === 0 && clip.status === 'done');
    localStorage.setItem(QUICK_CHAIN_KEY, JSON.stringify({
      chainId: chain.id, remoteFilename: opening?.remote_filename || result,
      prompt: opening?.prompt || prompt, negative, mode, audioMode, dialogue,
      width: resolution.width, height: resolution.height, duration,
      executionTarget: activeGenerationTarget, savedAt: Date.now(),
    }));
  }, [chain?.id, chain?.updated_at, result, prompt, negative, mode, audioMode, dialogue, resolution.width, resolution.height, duration, activeGenerationTarget]);

  useEffect(() => {
    if (!generating) return;
    const timer = window.setInterval(() => setElapsed((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [generating]);

  useEffect(() => {
    if (!continuing) return;
    const timer = window.setInterval(() => setContinuationElapsed((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [continuing]);

  useEffect(() => {
    if (!chain?.id || !continuing) return;
    const poll = window.setInterval(async () => {
      try {
        const value = await request(`/chain/${chain.id}`);
        setChain(value);
        const remoteActive = (value.clips || []).some((clip: Json) => clip.status === 'generating');
        if (remoteActive) setContinuationSeenRemote(true);
        if (!remoteActive && (continuationSeenRemote || Date.now() - continuationWatchStartedAt > 60000)) {
          setContinuing(false);
          const failed = [...(value.clips || [])].reverse().find((clip: Json) => clip.status === 'failed');
          if (failed?.metadata?.error) onError(failed.metadata.error);
        }
      } catch {
        // The original continuation request remains authoritative. A transient
        // recovery poll failure must not stop its visible elapsed timer.
      }
    }, 2500);
    return () => window.clearInterval(poll);
  }, [chain?.id, continuing, continuationSeenRemote, continuationWatchStartedAt, onError]);

  useEffect(() => {
    if (!promptId) return;
    const poll = window.setInterval(async () => {
      try {
        const value = await request(`/status/${promptId}`);
        if (value.status === 'done') {
          const filename = artifactName(value.files?.[0]);
          if (!filename) throw new Error('ComfyUI finished without a video file');
          localStorage.removeItem(QUICK_ACTIVE_KEY);
          setResult(filename); setGenerating(false); setPromptId('');
        } else if (value.status === 'error') {
          throw new Error(value.error || 'Generation failed');
        }
      } catch (cause) {
        localStorage.removeItem(QUICK_ACTIVE_KEY);
        setGenerating(false); setPromptId(''); onError(cause instanceof Error ? cause.message : 'Generation failed');
      }
    }, 2500);
    return () => window.clearInterval(poll);
  }, [promptId, onError]);

  useEffect(() => {
    if (!upscaleJob?.id || !ACTIVE_JOBS.has(upscaleJob.status)) return;
    const interval = window.setInterval(() => request(`/jobs/${upscaleJob.id}`).then(setUpscaleJob).catch((cause) => onError(cause.message)), 2000);
    return () => window.clearInterval(interval);
  }, [upscaleJob?.id, upscaleJob?.status, onError]);

  useEffect(() => {
    if (!upscaleJob?.id || !ACTIVE_JOBS.has(upscaleJob.status)) return;
    const interval = window.setInterval(() => setUpscaleElapsed((value) => value + 1), 1000);
    return () => window.clearInterval(interval);
  }, [upscaleJob?.id, upscaleJob?.status]);

  useEffect(() => {
    if (!prompt.trim()) {
      setCompilePreview(null);
      return;
    }
    const handle = window.setTimeout(() => {
      request('/generate/prompt-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          negative_prompt: negative,
          audio_mode: audioMode,
          duration_seconds: duration,
          dialogue: spokenWords,
        }),
      }).then(setCompilePreview).catch(() => {
        // Preview is advisory only; generation still uses the live compiler.
      });
    }, 320);
    return () => window.clearTimeout(handle);
  }, [prompt, negative, audioMode, duration, spokenWords]);

  async function generate() {
    if (!prompt.trim() || (mode === 'i2v' && !image)) return;
    setGenerating(true); setElapsed(0); setResult(''); setChain(null); setUpscaleJob(null); setActiveGenerationTarget(execution.generation); onError('');
    localStorage.removeItem(QUICK_CHAIN_KEY);
    try {
      let value;
      if (mode === 't2v') {
        const data = new FormData(); data.append('prompt', prompt); data.append('negative_prompt', negative); data.append('width', String(resolution.width)); data.append('height', String(resolution.height)); data.append('duration_seconds', String(duration)); data.append('profile', 'motion-draft-4x3'); data.append('audio_mode', audioMode); data.append('dialogue', spokenWords); data.append('execution_target', execution.generation);
        try {
          value = await request('/generate/t2v', { method: 'POST', body: data });
        } catch (cause) {
          // A still-running pre-recovery API accepts JSON only. Retry that one
          // validation mismatch without ever duplicating a queued render.
          if ((cause as Error & { status?: number }).status !== 422) throw cause;
          value = await request('/generate/t2v', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, negative_prompt: negative, width: resolution.width, height: resolution.height, duration_seconds: duration, profile: 'motion-draft-4x3', audio_mode: audioMode, dialogue: spokenWords, execution_target: execution.generation }) });
        }
      } else {
        const data = new FormData(); data.append('prompt', prompt); data.append('negative_prompt', negative); data.append('width', String(resolution.width)); data.append('height', String(resolution.height)); data.append('duration_seconds', String(duration)); data.append('image', image!); data.append('ic_lora_strength', '.5'); data.append('img_cond_strength', '.9'); data.append('audio_mode', audioMode); data.append('dialogue', spokenWords); data.append('execution_target', execution.generation);
        value = await request('/generate/i2v', { method: 'POST', body: data });
      }
      const active = {
        promptId: value.prompt_id, prompt, negative, mode, audioMode, dialogue, executionTarget: execution.generation,
        width: resolution.width, height: resolution.height, duration,
        startedAt: Date.now(),
      };
      localStorage.setItem(QUICK_ACTIVE_KEY, JSON.stringify(active));
      setPromptId(value.prompt_id);
    } catch (cause) { setGenerating(false); onError(cause instanceof Error ? cause.message : 'Generation failed'); }
  }

  async function loadRecent() {
    setLoadingRecent(true);
    try {
      const value = await request('/history?limit=12');
      setRecent(value.generations || []);
      setRecentChains(value.chains || []);
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : 'Could not load recent renders');
    } finally {
      setLoadingRecent(false);
    }
  }

  async function startChain() {
    try { applyChain(await request('/chain', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ video_file: result, prompt, execution_target: activeGenerationTarget }) })); }
    catch (cause) { onError(cause instanceof Error ? cause.message : 'Could not start chain'); }
  }

  async function continueChain() {
    if (!chain || !nextPrompt.trim()) return;
    const watchStartedAt = Date.now();
    setContinuationElapsed(0); setContinuing(true); setContinuationSeenRemote(false); setContinuationWatchStartedAt(watchStartedAt);
    const data = new FormData(); data.append('prompt', nextPrompt); data.append('strength', '.7'); data.append('width', String(resolution.width)); data.append('height', String(resolution.height)); data.append('duration_seconds', String(duration)); data.append('audio_mode', audioMode); data.append('dialogue', nextSpokenWords); data.append('execution_target', execution.continuation);
    try { const value = await request(`/chain/${chain.id}/continue`, { method: 'POST', body: data }); applyChain(value); setNextPrompt(''); setNextDialogue(''); }
    catch (cause) { onError(cause instanceof Error ? cause.message : 'Continuation failed'); }
    finally { setContinuing(false); setContinuationSeenRemote(false); }
  }

  async function rejectChainClip(clipId: string) {
    if (!chain || !window.confirm('Reject this continuation and return to the previous kept clip?')) return;
    try {
      setChain(await request(`/chain/${chain.id}/clips/${clipId}`, { method: 'DELETE' }));
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : 'Could not reject continuation');
    }
  }

  async function upscaleResult(scale: 2 | 4) {
    if (!result || !upscaleTarget?.available) return;
    try {
      setUpscaleScale(scale); setUpscaleElapsed(0);
      const value = await request('/upscales', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ remote_filename: result, target_id: execution.postUpscale, scale, prompt }) });
      setUpscaleJob(value.job);
    } catch (cause) { onError(cause instanceof Error ? cause.message : 'Upscale could not start'); }
  }

  async function upscaleMergedChain(scale: 2 | 4) {
    if (!chain?.id || chain.status !== 'merged' || !upscaleTarget?.available) return;
    try {
      setUpscaleScale(scale); setUpscaleElapsed(0);
      const value = await request('/upscales', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chain_id: chain.id,
          target_id: execution.postUpscale,
          scale,
        }),
      });
      setUpscaleJob(value.job);
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : 'Merged upscale could not start');
    }
  }

  async function recoverUpscaleResult() {
    if (!upscaleJob?.id) return;
    try {
      setUpscaleJob(await request(
        `/jobs/${upscaleJob.id}/recover-upscale`, { method: 'POST' }
      ));
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : 'Completed upscale could not be recovered');
    }
  }

  const completedChainClips = (chain?.clips || []).filter((clip: Json) => clip.status === 'done');
  const latestCompletedClip = completedChainClips[completedChainClips.length - 1];
  const latestFailedClip = [...(chain?.clips || [])].reverse().find((clip: Json) => clip.status === 'failed');
  const openingChainClip = completedChainClips.find((clip: Json) => clip.position === 0);
  const resultSource = restoringChain ? '' : (result ? outputUrl(result) : (openingChainClip && chain?.id ? `${API}/chain/${chain.id}/clips/${openingChainClip.id}/output` : ''));
  const resultDownload = result ? outputUrl(result) : resultSource;

  function reopenChain(value: Json) {
    applyChain(value);
    setGenerating(false); setPromptId(''); setRecent(null);
    localStorage.removeItem(QUICK_ACTIVE_KEY);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function closeChain() {
    setResult(openingChainClip?.remote_filename || '');
    setChain(null); setContinuing(false); setContinuationSeenRemote(false);
    localStorage.removeItem(QUICK_CHAIN_KEY);
  }

  return <main className="quick-main">
    <section className="quick-intro"><div><span className="eyebrow">Operator mode</span><h1>Direct LTX generation</h1><p>Author the visual prompt yourself and send it to the stable LTX 2.3 Q4_K_M GGUF workflow. No Ollama prompt processing; the selected audio contract is the only system-added direction.</p></div><button className="button secondary" disabled={loadingRecent} onClick={loadRecent}>{loadingRecent ? 'Checking workers…' : 'Recent renders'}</button></section>
    <div className="quick-layout">
      <section className="panel quick-form">
        <div className="segmented"><button className={mode === 't2v' ? 'active' : ''} onClick={() => setMode('t2v')}>Text to video</button><button className={mode === 'i2v' ? 'active' : ''} onClick={() => setMode('i2v')}>Image to video</button></div>
        <label>Human-authored visual prompt <span className="optional-label">never used as the dialogue field</span><textarea rows={8} value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Describe the complete shot over time: subject, movement, camera, setting, light, texture, and ending frame…" /></label>
        <label>Negative prompt<textarea rows={2} value={negative} onChange={(event) => setNegative(event.target.value)} placeholder="flicker, morphing, unstable identity, text…" /></label>
        {mode === 'i2v' && <label className="image-drop">Reference image<input type="file" accept="image/*" onChange={(event) => { const file = event.target.files?.[0] || null; setImage(file); setPreview(file ? URL.createObjectURL(file) : ''); }} />{preview && <img src={preview} alt="Reference preview" />}</label>}
        <div className="resolution-grid">{RESOLUTIONS.map((item) => <button key={item.label} className={resolution.label === item.label ? 'active' : ''} onClick={() => setResolution(item)}><strong>{item.label}</strong><span>{item.detail}</span></button>)}</div>
        <div className="quick-options"><label>Clip length<select value={duration} onChange={(event) => setDuration(Number(event.target.value))}><option value={2}>2s · pipeline check</option><option value={5}>5s · standard</option><option value={8}>8s</option><option value={10}>10s</option></select></label><label>Generated audio<select value={audioMode} onChange={(event) => setAudioMode(event.target.value as AudioMode)}><option value="ambient">Ambience + Foley · sound-on default</option><option value="silent">Silent · guaranteed no voice</option><option value="native-dialogue">Native quoted speech · generative</option><option value="prompt">Prompt-controlled speech · advanced</option></select></label><div><strong>{generationMachine}</strong><span>GGUF · motion-first 4 + 3</span></div></div>
        {audioMode === 'silent' && <p className="quick-audio-note">Guaranteed no-voice mode: LTX still renders the picture, but its generated audio is disconnected before the MP4 is saved.</p>}
        {audioMode === 'ambient' && <p className="quick-audio-note">LTX will generate environmental sound and Foley with a zero-speech prompt guard. Because audio and video are generated jointly, only Silent can guarantee that no voice appears.</p>}
        {audioMode === 'native-dialogue' && <div className="quick-dialogue"><label>Only words to speak<textarea rows={2} value={dialogue} onChange={(event) => setDialogue(event.target.value)} placeholder={quotedWords(prompt) || 'Enter the exact short line without stage directions…'} /></label><p className="quick-audio-note">{dialogue.trim() ? 'This separate field is the dialogue source.' : quotedWords(prompt) ? `Detected from quotation marks: “${quotedWords(prompt)}”` : 'Add the spoken line here or put it in quotation marks in the visual prompt.'} Native LTX speech is generative; an uploaded voiceover in Finish social cut is the guaranteed verbatim option. Budget: about {Math.max(3, Math.floor(duration * 2.5))} words.</p></div>}
        {looksLikeScript && <p className="quick-script-hint"><strong>Multi-shot script detected</strong><span>{scriptShape?.hint || 'Split this in Studio → Script to video, or paste one visual beat for Quick Generate. Comedy dialogue is most reliable as Finish social cut voiceover.'}</span>{Array.isArray(scriptShape?.reasons) && scriptShape.reasons.length > 0 && <small>{scriptShape.reasons.join(' · ')}</small>}</p>}
        {prompt.trim() && compilePreview && <div className={`ltx-preview ${showLtxPreview ? 'open' : ''} ${compilePreview.ok === false ? 'ltx-preview-error' : looksLikeScript ? 'ltx-preview-warn' : ''}`}>
          <button type="button" className="ltx-preview-toggle" onClick={() => setShowLtxPreview((value) => !value)}>
            <span>Sending to LTX · {duration}s · {audioMode === 'native-dialogue' ? 'Native dialogue' : audioMode === 'silent' ? 'Silent' : audioMode === 'prompt' ? 'Prompt speech' : 'Ambience'}</span>
            <strong>{showLtxPreview ? 'Hide' : 'Show'}</strong>
          </button>
          {compilePreview.suggested_audio_mode === 'native-dialogue' && compilePreview.detected_dialogue && <div className="ltx-preview-audio-fix"><span>Dialogue detected: “{compilePreview.detected_dialogue}”</span><button type="button" className="button secondary" onClick={() => { setDialogue(compilePreview.detected_dialogue); setAudioMode('native-dialogue'); }}>Use Native dialogue</button></div>}
          {showLtxPreview && <div className="ltx-preview-body">
            <p className="ltx-preview-summary">{compilePreview.summary}</p>
            {compilePreview.error && <p className="ltx-preview-error-text">{compilePreview.error}</p>}
            <label>Visual after compile<textarea rows={4} readOnly value={compilePreview.visual_prompt || ''} /></label>
            {compilePreview.dialogue && <label>Spoken line<input readOnly value={compilePreview.dialogue} />{typeof compilePreview.dialogue_words === 'number' && <small>{compilePreview.dialogue_words}/{compilePreview.word_limit} words</small>}</label>}
            {compilePreview.compiled_prompt && <label>Full AV contract<textarea rows={3} readOnly value={compilePreview.compiled_prompt} /></label>}
          </div>}
        </div>}
        <button className="button primary generate-direct" disabled={generating || !prompt.trim() || (mode === 'i2v' && !image) || (audioMode === 'native-dialogue' && !spokenWords) || compilePreview?.ok === false} onClick={generate}>{generating ? `Generating · ${elapsed}s` : 'Generate with LTX 2.3'}</button>
      </section>
      <section className="panel quick-result">
        {!resultSource && !generating && !restoringChain && <EmptyState title="Your render appears here">The default is a real 5-second clip. The 2-second option is labeled and reserved for pipeline checks.</EmptyState>}
        {restoringChain && <div className="rendering"><div className="render-orbit"><i /><span>↻</span></div><h2>Reopening your story</h2><p>Loading the latest persisted continuity chain…</p></div>}
        {generating && <div className="rendering"><div className="render-orbit"><i /><span>{elapsed}s</span></div><h2>{generationMachine} is rendering</h2><p>Prompt queued as {promptId ? promptId.slice(0, 12) : 'preflight'}…</p><div className="render-progress"><i /></div></div>}
        {resultSource && <div className="direct-result"><video controls autoPlay loop src={resultSource} /><div className="result-title"><div><span className="eyebrow">Generation complete · {generationMachine}</span><h2>{duration}s GGUF clip</h2></div><a className="button success" href={resultDownload} download>Download</a></div>{result && <div className="result-processing"><button className="button secondary" disabled={!upscaleTarget?.available || Boolean(upscaleJob && ACTIVE_JOBS.has(upscaleJob.status))} title={upscaleTarget?.unavailable_reason || ''} onClick={() => upscaleResult(2)}>{upscaleJob && ACTIVE_JOBS.has(upscaleJob.status) ? `Pixel ${upscaleScale}× · ${Math.round(Number(upscaleJob.progress || 0) * 100)}% · ${upscaleElapsed}s` : `Pixel 2× · ${upscaleTarget?.label || execution.postUpscale}`}</button><button className="button secondary" disabled={!upscaleTarget?.available || Boolean(upscaleJob && ACTIVE_JOBS.has(upscaleJob.status))} title="More generative x4 mode; output is capped for 24 GB VRAM" onClick={() => upscaleResult(4)}>Pixel 4×</button>{upscaleJob?.status === 'succeeded' && <a className="button success" href={`${API}/jobs/${upscaleJob.id}/output`}>Download {upscaleScale}×</a>}{upscaleJob?.status === 'failed' && <span>{upscaleJob.error || 'Upscale failed'}</span>}</div>}{!chain ? <button className="button secondary wide" onClick={startChain}>Continue this clip</button> : <div className="chain-box"><div className="chain-title"><h3>Continuity chain · {completedChainClips.length} kept clip{completedChainClips.length === 1 ? '' : 's'}</h3><button className="text-button" onClick={closeChain}>Close story</button></div>{completedChainClips.filter((clip: Json) => clip.position > 0).length > 0 && <div className="chain-previews">{completedChainClips.filter((clip: Json) => clip.position > 0).map((clip: Json) => <article key={clip.id} className="chain-preview"><div><span>Clip {clip.position + 1}</span><strong>{clip.prompt}</strong></div><ChainClipVideo chainId={chain.id} clip={clip} />{latestCompletedClip?.id === clip.id && <button className="button reject" onClick={() => rejectChainClip(clip.id)}>Reject & redo</button>}</article>)}</div>}{latestFailedClip?.metadata?.error && <div className="chain-failure"><strong>Last continuation failed</strong><span>{latestFailedClip.metadata.error}</span></div>}<textarea rows={3} value={nextPrompt} onChange={(event) => setNextPrompt(event.target.value)} placeholder="What happens next? The previous ending frame becomes the continuity anchor…" />{audioMode === 'native-dialogue' && <div className="quick-dialogue compact"><label>Only words for this next clip<textarea rows={2} value={nextDialogue} onChange={(event) => setNextDialogue(event.target.value)} placeholder={quotedWords(nextPrompt) || 'Enter only the next spoken line…'} /></label></div>}<button className="button primary" disabled={continuing || !nextPrompt.trim() || (audioMode === 'native-dialogue' && !nextSpokenWords)} onClick={continueChain}>{continuing ? `Rendering continuation · ${continuationElapsed}s` : 'Add next clip'}</button>{continuing && <div className="continuation-progress"><i /><span>{continuationMachine} is extending the story from the previous ending frame. You can switch tabs or reload; this chain is persisted.</span></div>}{completedChainClips.length >= 2 && <button className="button secondary" disabled={continuing} onClick={async () => { try { const value = await request(`/chain/${chain.id}/merge`, { method: 'POST' }); setChain(value.chain); } catch (cause) { onError(cause instanceof Error ? cause.message : 'Merge failed'); } }}>Merge kept clips</button>}{chain.status === 'merged' && <div className="merged-chain"><video controls preload="metadata" src={`${API}/chain/${chain.id}/output`} /><div><a className="button success" href={`${API}/chain/${chain.id}/output`}>Download merged video</a><button className="button secondary" disabled={!upscaleTarget?.available || Boolean(upscaleJob && ACTIVE_JOBS.has(upscaleJob.status))} onClick={() => upscaleMergedChain(2)}>{upscaleJob && ACTIVE_JOBS.has(upscaleJob.status) ? `Upscaling merged video · ${Math.round(Number(upscaleJob.progress || 0) * 100)}% · ${upscaleElapsed}s` : 'Pixel 2× merged video'}</button><button className="button secondary" disabled={!upscaleTarget?.available || Boolean(upscaleJob && ACTIVE_JOBS.has(upscaleJob.status))} onClick={() => upscaleMergedChain(4)}>Pixel 4× merged video</button>{upscaleJob?.status === 'succeeded' && <a className="button success" href={`${API}/jobs/${upscaleJob.id}/output`}>Download upscaled merged video</a>}{upscaleJob?.status === 'failed' && <span>{upscaleJob.error || 'Merged upscale failed'}</span>}</div></div>}</div>}</div>}
        {upscaleJob?.status === 'failed' && <button className="button secondary wide" onClick={recoverUpscaleResult}>Recover completed upscale + restore audio</button>}
      </section>
    </div>
    {recent && <section className="panel recent-renders"><div className="section-head"><div><span className="eyebrow">Recovery</span><h2>Recent renders and stories</h2><p>Recover a completed take or reopen a persisted multi-clip continuity chain.</p></div><button className="button ghost" onClick={loadRecent}>Refresh</button></div>{recentChains.length > 0 && <div className="recent-chain-grid">{recentChains.map((item: Json) => { const kept = (item.clips || []).filter((clip: Json) => clip.status === 'done'); const active = (item.clips || []).some((clip: Json) => clip.status === 'generating'); return <button key={item.id} onClick={() => reopenChain(item)}><span>{active ? 'Rendering' : item.status === 'merged' ? 'Merged story' : 'Continuity story'}</span><strong>{kept[0]?.prompt || 'Untitled continuity chain'}</strong><small>{kept.length} kept clip{kept.length === 1 ? '' : 's'} · {relativeTime(item.updated_at)}</small></button>; })}</div>}<div className="recent-grid">{recent.map((item) => {
      const filename = artifactName(item.files?.[0]);
      return <button key={item.prompt_id} disabled={!filename} onClick={() => { setResult(filename); setPrompt(item.prompt || prompt); setGenerating(false); setPromptId(''); localStorage.removeItem(QUICK_ACTIVE_KEY); }}><span>{item.elapsed_seconds ? `${Math.round(item.elapsed_seconds)}s render` : prettyStatus(item.status)}</span><strong>{item.prompt || filename || 'Completed generation'}</strong><small>{filename}</small></button>;
    })}{!recent.length && <p>No completed ComfyUI renders were found yet. Refresh after the active server job finishes.</p>}</div></section>}
    {chain && <QuickFinish chain={chain} resolution={resolution} onChain={applyChain} onError={onError} />}
  </main>;
}

export default App;
