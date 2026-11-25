"""CLI using Typer."""

import typer

app = typer.Typer(name="CLI using Typer")


@app.command()
def say_hi(name):
    """Say hello to the user."""
    typer.echo(f"Hello: {name}")


@app.command()
def goodbye():
    """Say goodbye to the user."""
    typer.echo("Goodbye!")


@app.command()
def greet(name):
    """Greet the user."""
    typer.echo(f"Greetings, {name}!")


@app.command()
def welcome(name, greeting: str = "Welcome", enthousiastic: bool = False):
    """Greet the user."""
    typer.echo(f"{greeting}, {name}{'!' if enthousiastic else ''}")


if __name__ == "__main__":
    app()
