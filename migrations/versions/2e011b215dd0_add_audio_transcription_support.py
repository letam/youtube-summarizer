"""Add audio transcription support

Revision ID: 2e011b215dd0
Revises: 
Create Date: 2026-01-26 16:22:27.790525

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2e011b215dd0'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add new columns (nullable initially to allow data migration)
    with op.batch_alter_table('transcript', schema=None) as batch_op:
        batch_op.add_column(sa.Column('source_type', sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column('source_id', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('original_filename', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('file_path', sa.String(length=500), nullable=True))
        batch_op.add_column(sa.Column('audio_duration_seconds', sa.Integer(), nullable=True))

    # Migrate data: copy video_id to source_id, set source_type to 'youtube'
    op.execute("UPDATE transcript SET source_id = video_id, source_type = 'youtube'")

    # Now make columns not nullable and add constraints
    with op.batch_alter_table('transcript', schema=None) as batch_op:
        batch_op.alter_column('source_type', nullable=False)
        batch_op.alter_column('source_id', nullable=False)
        batch_op.create_unique_constraint('uq_transcript_source_id', ['source_id'])
        batch_op.drop_column('video_id')


def downgrade():
    # Add video_id column back
    with op.batch_alter_table('transcript', schema=None) as batch_op:
        batch_op.add_column(sa.Column('video_id', sa.VARCHAR(length=11), nullable=True))

    # Migrate data back: copy source_id to video_id (only for youtube type)
    op.execute("UPDATE transcript SET video_id = source_id WHERE source_type = 'youtube'")

    # Remove audio records (can't downgrade those)
    op.execute("DELETE FROM transcript WHERE source_type = 'audio'")

    # Make video_id not nullable and drop new columns
    with op.batch_alter_table('transcript', schema=None) as batch_op:
        batch_op.alter_column('video_id', nullable=False)
        batch_op.drop_constraint('uq_transcript_source_id', type_='unique')
        batch_op.drop_column('audio_duration_seconds')
        batch_op.drop_column('file_path')
        batch_op.drop_column('original_filename')
        batch_op.drop_column('source_id')
        batch_op.drop_column('source_type')
