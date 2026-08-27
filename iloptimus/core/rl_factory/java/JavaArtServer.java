import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import javax.imageio.ImageIO;
import javax.tools.*;

/**
 * JavaArtServer — persistent JVM for ultra-fast in-memory Java code compilation
 * and execution of art-painting code.
 *
 * Protocol (binary, over stdin/stdout):
 *   Request:  [4-byte BE length] [UTF-8 source code]
 *   Response: [1-byte status] [4-byte BE length] [payload]
 *     status 0 = success  → payload = PNG bytes
 *     status 1 = compile error → payload = UTF-8 error text
 *     status 2 = runtime error → payload = UTF-8 error text
 *     status 3 = timeout → payload = UTF-8 "timeout"
 *
 * The source code must define a class with a static method:
 *   public static BufferedImage paint(int width, int height)
 *
 * The class name is auto-detected from the source. The server calls
 * paint(canvasWidth, canvasHeight) and encodes the result as PNG.
 *
 * Security: runs in a thread with a hard timeout. No SecurityManager
 * (deprecated in Java 17+); instead we rely on subprocess isolation and
 * timeout. The server process itself is disposable — kill and restart
 * if it misbehaves.
 */
public class JavaArtServer {

    private static final int DEFAULT_WIDTH = 512;
    private static final int DEFAULT_HEIGHT = 512;
    private static final long TIMEOUT_MS = 10_000;

    private final JavaCompiler compiler;
    private final InputStream in;
    private final OutputStream out;

    public JavaArtServer() {
        this.compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            throw new RuntimeException("No Java compiler available — need JDK, not JRE");
        }
        this.in = new BufferedInputStream(System.in);
        this.out = new BufferedOutputStream(System.out);
    }

    /** Create a fresh file manager + class loader for each compilation. */
    private InMemoryFileManager newFileManager() {
        return new InMemoryFileManager(
            compiler.getStandardFileManager(null, null, StandardCharsets.UTF_8)
        );
    }

    public void run() throws IOException {
        while (true) {
            byte[] sourceBytes;
            try {
                sourceBytes = readLengthPrefixed(in);
            } catch (EOFException e) {
                break; // stdin closed, exit
            }
            if (sourceBytes == null || sourceBytes.length == 0) {
                sendResponse((byte) 1, "empty source".getBytes(StandardCharsets.UTF_8));
                continue;
            }

            String source = new String(sourceBytes, StandardCharsets.UTF_8);
            processSource(source);
        }
    }

    private void processSource(String source) {
        // Detect class name from the source
        String className = detectClassName(source);
        if (className == null) {
            sendResponse((byte) 1, "cannot find public class name in source".getBytes(StandardCharsets.UTF_8));
            return;
        }

        // Fresh file manager per compilation — prevents class caching bugs
        InMemoryFileManager fm = newFileManager();

        // Compile in-memory
        InMemoryJavaSource sourceObj = new InMemoryJavaSource(className, source);
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();

        boolean compiled;
        try {
            compiled = compiler.getTask(null, fm, diagnostics,
                    null, null, List.of(sourceObj)).call();
        } catch (Exception e) {
            sendResponse((byte) 1, ("compiler exception: " + e.getMessage()).getBytes(StandardCharsets.UTF_8));
            return;
        }

        if (!compiled) {
            StringBuilder sb = new StringBuilder();
            for (Diagnostic<?> d : diagnostics.getDiagnostics()) {
                sb.append(d.toString()).append('\n');
            }
            sendResponse((byte) 1, sb.toString().getBytes(StandardCharsets.UTF_8));
            return;
        }

        // Execute in a thread with timeout
        final InMemoryFileManager execFm = fm;
        ExecutorService exec = Executors.newSingleThreadExecutor();
        Future<byte[]> future = exec.submit(() -> executePaint(className, execFm));

        byte[] pngBytes;
        try {
            pngBytes = future.get(TIMEOUT_MS, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            exec.shutdownNow();
            sendResponse((byte) 3, "timeout".getBytes(StandardCharsets.UTF_8));
            return;
        } catch (ExecutionException e) {
            sendResponse((byte) 2, ("runtime error: " + e.getCause().getMessage()).getBytes(StandardCharsets.UTF_8));
            exec.shutdownNow();
            return;
        } catch (InterruptedException e) {
            sendResponse((byte) 2, "interrupted".getBytes(StandardCharsets.UTF_8));
            exec.shutdownNow();
            return;
        }
        exec.shutdownNow();

        if (pngBytes == null) {
            sendResponse((byte) 2, "paint() returned null".getBytes(StandardCharsets.UTF_8));
            return;
        }

        sendResponse((byte) 0, pngBytes);
    }

    @SuppressWarnings("unchecked")
    private byte[] executePaint(String className, InMemoryFileManager fm) {
        try {
            Class<?> clazz = fm.getClassLoader().loadClass(className);
            java.lang.reflect.Method method = clazz.getMethod("paint", int.class, int.class);
            Object result = method.invoke(null, DEFAULT_WIDTH, DEFAULT_HEIGHT);
            if (!(result instanceof BufferedImage)) {
                return null;
            }
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write((BufferedImage) result, "PNG", baos);
            return baos.toByteArray();
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage(), e);
        }
    }

    // --- Protocol I/O ---

    private static byte[] readLengthPrefixed(InputStream in) throws IOException {
        byte[] lenBuf = new byte[4];
        int read = 0;
        while (read < 4) {
            int n = in.read(lenBuf, read, 4 - read);
            if (n == -1) {
                if (read == 0) throw new EOFException();
                throw new IOException("unexpected EOF reading length");
            }
            read += n;
        }
        int length = ByteBuffer.wrap(lenBuf).getInt();
        if (length < 0 || length > 10_000_000) {
            throw new IOException("invalid length: " + length);
        }
        byte[] data = new byte[length];
        read = 0;
        while (read < length) {
            int n = in.read(data, read, length - read);
            if (n == -1) throw new IOException("unexpected EOF reading body");
            read += n;
        }
        return data;
    }

    private void sendResponse(byte status, byte[] payload) {
        try {
            synchronized (out) {
                out.write(status);
                out.write(ByteBuffer.allocate(4).putInt(payload.length).array());
                out.write(payload);
                out.flush();
            }
        } catch (IOException e) {
            // stdout broken, can't do anything
            System.exit(1);
        }
    }

    // --- Class name detection ---

    private static String detectClassName(String source) {
        // Match: public class <Name>
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(
            "public\\s+(?:final\\s+)?class\\s+(\\w+)"
        );
        java.util.regex.Matcher m = p.matcher(source);
        if (m.find()) return m.group(1);
        // Fallback: any class
        p = java.util.regex.Pattern.compile("class\\s+(\\w+)");
        m = p.matcher(source);
        if (m.find()) return m.group(1);
        return null;
    }

    // --- In-memory compilation support ---

    /** A Java source string wrapped as a JavaFileObject. */
    static class InMemoryJavaSource extends SimpleJavaFileObject {
        private final String source;

        InMemoryJavaSource(String className, String source) {
            super(java.net.URI.create("string:///" + className.replace('.', '/') + ".java"),
                  Kind.SOURCE);
            this.source = source;
        }

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            return source;
        }
    }

    /** A compiled class stored as a byte array. */
    static class InMemoryClassOutput extends SimpleJavaFileObject {
        private final String className;
        private byte[] bytes;

        InMemoryClassOutput(String className) {
            super(java.net.URI.create("bytes:///" + className.replace('.', '/') + ".class"),
                  Kind.CLASS);
            this.className = className;
        }

        @Override
        public OutputStream openOutputStream() {
            return new ByteArrayOutputStream() {
                @Override
                public void close() throws IOException {
                    super.close();
                    bytes = toByteArray();
                }
            };
        }

        byte[] getBytes() { return bytes; }
        String getClassName() { return className; }
    }

    /** File manager that stores compiled classes in memory. */
    static class InMemoryFileManager extends ForwardingJavaFileManager<JavaFileManager> {
        private final Map<String, InMemoryClassOutput> classes = new HashMap<>();
        private final InMemoryClassLoader classLoader = new InMemoryClassLoader(classes);

        InMemoryFileManager(JavaFileManager standard) {
            super(standard);
        }

        @Override
        public JavaFileObject getJavaFileForOutput(Location location, String className,
                                                     JavaFileObject.Kind kind, FileObject sibling) {
            InMemoryClassOutput output = new InMemoryClassOutput(className);
            classes.put(className, output);
            return output;
        }

        @Override
        public ClassLoader getClassLoader(Location location) {
            return classLoader;
        }

        InMemoryClassLoader getClassLoader() { return classLoader; }
    }

    /** Classloader that loads from the in-memory class map. */
    static class InMemoryClassLoader extends ClassLoader {
        private final Map<String, InMemoryClassOutput> classes;

        InMemoryClassLoader(Map<String, InMemoryClassOutput> classes) {
            super(Thread.currentThread().getContextClassLoader());
            this.classes = classes;
        }

        @Override
        protected Class<?> findClass(String name) throws ClassNotFoundException {
            InMemoryClassOutput output = classes.get(name);
            if (output != null && output.getBytes() != null) {
                return defineClass(name, output.getBytes(), 0, output.getBytes().length);
            }
            return super.findClass(name);
        }
    }

    // --- Main ---

    public static void main(String[] args) throws Exception {
        // Reduce ImageIO overhead: don't scan for plugins
        System.setProperty("java.awt.headless", "true");
        ImageIO.setUseCache(false);

        JavaArtServer server = new JavaArtServer();
        server.run();
    }
}
