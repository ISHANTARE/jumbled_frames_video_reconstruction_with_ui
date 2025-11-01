from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import uuid
from video_reconstruction import VideoReconstructor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Generate unique filename
        file_id = str(uuid.uuid4())
        input_path = f"temp_{file_id}_input.mp4"
        output_path = f"temp_{file_id}_output.mp4"

        # Save uploaded file
        video_file.save(input_path)

        # Reconstruct video
        reconstructor = VideoReconstructor()
        success, result = reconstructor.reconstruct_video(input_path, output_path)

        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)

        if success:
            return jsonify({
                'success': True,
                'execution_time': result['execution_time'],
                'frame_count': result['frame_count'],
                'output_file': output_path
            })
        else:
            # Clean up output file if it was created but failed
            if os.path.exists(output_path):
                os.remove(output_path)
            return jsonify({'error': result}), 500

    except Exception as e:
        # Clean up on error
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    try:
        if filename.startswith('temp_') and filename.endswith('_output.mp4'):
            return send_file(filename, as_attachment=True, download_name='reconstructed_video.mp4')
        else:
            return jsonify({'error': 'Invalid file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cleanup/<filename>')
def cleanup_file(filename):
    """Clean up temporary files"""
    try:
        if filename.startswith('temp_') and os.path.exists(filename):
            os.remove(filename)
        return jsonify({'success': True})
    except:
        return jsonify({'success': False})


if __name__ == '__main__':
    # Create directories if they don't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, host='0.0.0.0', port=5000)