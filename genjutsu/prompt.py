from .controls import ControlPackage, Operation

def compile_prompt(controls: ControlPackage) -> str:
    operation={Operation.MOTION_TRANSFER:"Transfer the original motion to the referenced subject.",Operation.CHARACTER_SWAP:"Replace the selected character while preserving its performance.",Operation.OBJECT_SWAP:"Replace only the selected object with the supplied reference.",Operation.PRODUCT_SWAP:"Replace only the selected product while preserving the advertisement.",Operation.OUTFIT_SWAP:"Replace the selected clothing while preserving the wearer and motion.",Operation.ENVIRONMENT_RECAST:"Rebuild the environment while preserving the original subject motion and camera.",Operation.FULL_RECAST:"Reinterpret the scene using the references while preserving the source timing and motion.",Operation.STYLE_RECAST:"Apply the requested visual style while preserving scene composition and motion."}[controls.operation]
    refs=", ".join(f"{r.category.lower()} reference (weight {r.weight:g})" for r in controls.references.references)
    invariants=controls.settings.get("preserve",["motion","camera","composition"])
    lines=[operation]
    if controls.prompt.strip(): lines.append(controls.prompt.strip())
    if refs: lines.append(f"Use the provided {refs}.")
    lines.append("Preserve: "+", ".join(invariants)+".")
    if controls.negative_prompt.strip(): lines.append("Avoid: "+controls.negative_prompt.strip())
    return "\n\n".join(lines)
